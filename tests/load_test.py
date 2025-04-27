"""
Load testing for inference manager API using Locust.
Simulates concurrent users making requests to all endpoints.
"""
from locust import HttpUser, task, between
import random
import json

class InferenceManagerUser(HttpUser):
    wait_time = between(1, 5)  # Random wait between requests
    
    models = ["gpt-4", "llama-2", "stable-diffusion"]
    text_prompts = [
        "Explain quantum computing",
        "Write a poem about AI",
        "Summarize this text"
    ]
    image_prompts = [
        "A futuristic city skyline",
        "A cat wearing sunglasses",
        "Abstract art in blue tones"
    ]
    
    @task(3)
    def test_txt2txt(self):
        """Text-to-text generation test"""
        payload = {
            "model": random.choice(self.models),
            "messages": [{"role": "user", "content": random.choice(self.text_prompts)}],
            "temperature": random.uniform(0.5, 1.0)
        }
        self.client.post("/tools/txt2txt", json=payload)
    
    @task(2)
    def test_txt2img(self):
        """Text-to-image generation test"""
        payload = {
            "model": "stable-diffusion",
            "prompt": random.choice(self.image_prompts),
            "height": random.choice([256, 512, 768]),
            "width": random.choice([256, 512, 768])
        }
        self.client.post("/tools/txt2img", json=payload)
    
    @task(1)
    def test_img2img(self):
        """Image-to-image generation test"""
        payload = {
            "model": "stable-diffusion",
            "file": "example.png",  # In real test would use actual image data
            "prompt": random.choice(self.image_prompts),
            "height": 512,
            "width": 512
        }
        self.client.post("/tools/img2img", json=payload)
    
    @task(1)
    def test_img2txt(self):
        """Image-to-text generation test"""
        payload = {
            "model": random.choice(self.models),
            "messages": [{"role": "user", "content": "Describe this image"}],
            "temperature": 0.7
        }
        self.client.post("/tools/img2txt", json=payload)
    
    @task(1)
    def test_get_metrics(self):
        """Metrics endpoint test"""
        self.client.get("/tools/get_request_metrics")
        self.client.get("/tools/get_error_metrics")

class PeakTrafficUser(InferenceManagerUser):
    """Simulates peak traffic conditions with shorter wait times"""
    wait_time = between(0.1, 1)

class SustainedLoadUser(InferenceManagerUser):
    """Simulates sustained load with longer test duration"""
    wait_time = between(5, 10)