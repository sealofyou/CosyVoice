#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import websockets
import json
import base64
import numpy as np
import sounddevice as sd
from collections import deque
import threading
import queue

class AudioStreamer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.playback_finished = threading.Event()
        self.stream = None
        
    def start_playback(self, sample_rate):
        """启动音频播放流"""
        self.stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype=np.float32)
        self.stream.start()
        
    def play_chunk(self, audio_data):
        """播放音频块"""
        if self.stream is not None:
            # 确保数据是一维的
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
            self.stream.write(audio_data)
            
    def stop_playback(self):
        """停止音频播放"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

async def test_websocket_streaming():
    uri = "ws://localhost:8001"  # 根据实际情况调整端口号
    streamer = AudioStreamer()
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket server")
            
            # 准备测试数据
            test_request = {
                "tts_text": "你好，这是一个WebSocket流式测试。我们将逐步接收并播放音频数据，而不是等待全部接收完毕。",
                "mode": "3s极速复刻",
                "prompt_text": "希望你以后做的比我还好哦",
                "prompt_wav": "./asset/zero_shot_prompt.wav",
                "stream": True,
                "speed": 1.0,
                "seed": 12345
            }
            
            # 发送请求
            await websocket.send(json.dumps(test_request))
            print("Sent TTS request")
            
            all_audio_data = []
            sample_rate = 24000  # 默认采样率
            
            # 接收响应
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data["type"] == "start":
                        print("Start receiving audio data...")
                        
                    elif data["type"] == "audio":
                        # 获取采样率
                        sample_rate = data["sample_rate"]
                        
                        # 解码音频数据
                        audio_bytes = base64.b64decode(data["audio_data"])
                        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                        all_audio_data.append(audio_array)
                        
                        # 如果是第一段音频，启动播放流
                        if len(all_audio_data) == 1:
                            streamer.start_playback(sample_rate)
                        
                        # 实时播放音频块
                        streamer.play_chunk(audio_array)
                        print(f"Played audio chunk with {len(audio_array)} samples")
                        
                    elif data["type"] == "end":
                        print("Finished receiving audio data")
                        break
                        
                    elif data["type"] == "error":
                        print(f"Error: {data['message']}")
                        break
                        
                except json.JSONDecodeError:
                    print("Received non-JSON message")
            
            # 停止播放
            streamer.stop_playback()
            
            # 保存完整音频到文件供参考
            if all_audio_data:
                full_audio = np.concatenate(all_audio_data)
                import scipy.io.wavfile as wavfile
                wavfile.write("output_full.wav", sample_rate, full_audio.astype(np.float32))
                print(f"Full audio saved to output_full.wav ({len(full_audio)} samples)")
                
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")
        streamer.stop_playback()
    except Exception as e:
        print(f"Error: {e}")
        streamer.stop_playback()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket_streaming())