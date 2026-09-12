"""Lazy landmark recognition. A fine-tuned checkpoint can be plugged in via secrets/config."""
import base64
from io import BytesIO
import json
import re
from functools import lru_cache
from typing import Any
import requests

LANDMARK_HINTS = {"hunza": "Hunza", "skardu": "Skardu", "lahore": "Lahore", "islamabad": "Islamabad"}
LOCAL_CANDIDATES = [
    "Minar-e-Pakistan Lahore", "Faisal Mosque Islamabad", "Badshahi Mosque Lahore", "Lahore Fort Pakistan",
    "Mazar-e-Quaid Karachi", "Pakistan Monument Islamabad", "Red Fort Delhi India", "Red Fort Lahore Pakistan",
    "Jama Masjid Delhi India", "Humayun's Tomb Delhi India",
    "Hunza Valley Pakistan", "Attabad Lake Pakistan",
    "Baltit Fort Pakistan", "Skardu Pakistan", "Deosai National Park Pakistan", "Malam Jabba Swat Pakistan",
    "Naran Kaghan Pakistan", "Chitral Pakistan", "a natural mountain landscape", "a city street", "a mosque or religious building",
    "a historic building", "a modern building", "food or a restaurant dish", "a person or group of people",
    "an animal", "a vehicle", "an indoor room", "a beach or coastline", "an unknown place",
    "Machu Picchu Peru", "Eiffel Tower Paris France", "Taj Mahal Agra India", "Statue of Liberty New York",
    "Big Ben London England", "Colosseum Rome Italy", "Great Wall of China", "Petra Jordan",
    "Burj Khalifa Dubai", "Sagrada Familia Barcelona Spain", "Pyramids of Giza Egypt",
]
LANDMARK_DESCRIPTIONS = {
    "Red Fort Delhi India": "the Red Fort in Delhi, a massive red sandstone Mughal fort with long red walls, bastions, arches, and small domed pavilions",
    "Red Fort Lahore Pakistan": "the Lahore Fort, a red sandstone Mughal fort with large walls, arches, and historic courtyards",
    "Taj Mahal Agra India": "the Taj Mahal, a white marble mausoleum with one large white onion dome and four minarets",
    "Mazar-e-Quaid Karachi": "Mazar-e-Quaid, a bright white marble mausoleum with a large white dome and broad empty plaza",
    "Badshahi Mosque Lahore": "Badshahi Mosque Lahore, a huge red sandstone mosque with three white domes and tall minarets",
    "Faisal Mosque Islamabad": "Faisal Mosque Islamabad, a large white tent-shaped modern mosque surrounded by hills",
    "Minar-e-Pakistan Lahore": "Minar-e-Pakistan, a tall white concrete tower with a flared flower-like base",
}


@lru_cache(maxsize=1)
def _load_local_vision_model():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        from transformers.models.clip import CLIPModel, CLIPProcessor
    model_name = "openai/clip-vit-base-patch32"
    return CLIPProcessor.from_pretrained(model_name), CLIPModel.from_pretrained(model_name)


def _recognize_locally(image: Any) -> dict[str, Any]:
    processor, model = _load_local_vision_model()
    import torch
    landmark_candidates = [candidate for candidate in LOCAL_CANDIDATES if candidate in LANDMARK_DESCRIPTIONS]
    landmark_candidates += [candidate for candidate in LOCAL_CANDIDATES if candidate not in LANDMARK_DESCRIPTIONS and not candidate.startswith("a ")]
    prompt_templates = ["a photo of {}", "a travel photo showing {}", "a photograph of the architecture of {}", "a historic landmark identified as {}"]
    prompt_scores = []
    with torch.no_grad():
        for template in prompt_templates:
            texts = [template.format(LANDMARK_DESCRIPTIONS.get(candidate, candidate)) for candidate in landmark_candidates]
            inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
            prompt_scores.append(model(**inputs).logits_per_image.softmax(dim=1)[0])
        probabilities = torch.stack(prompt_scores).mean(dim=0)
    confidence, index = probabilities.max(dim=0)
    second_best = probabilities.topk(min(2, len(landmark_candidates))).values[-1]
    confidence_value = min(float(confidence), max(0.55, min(0.93, float(confidence + (confidence - second_best) * 0.5))))
    label = landmark_candidates[int(index)]
    if label == "an unknown place":
        label = "Unknown landmark"
    return {"landmark": label, "title": label, "category": "visual match", "description": f"The image most closely matches {label}. This is a visual category match, not a guaranteed exact landmark identification.", "confidence": confidence_value, "low_confidence": confidence_value < 0.7, "message": "Analyzed with local CLIP visual matching; verify the result if confidence is below 70%."}


def recognize_landmark(image_bytes: bytes, api_key: str | None = None, model: str | None = None) -> dict[str, Any]:
    """Describe any uploaded image and identify a landmark when one is visible."""
    try:
        from PIL import Image
        image = Image.open(BytesIO(image_bytes))
        if image.width < 20 or image.height < 20:
            return {"landmark": "Unknown", "confidence": 0.0, "low_confidence": True, "message": "Image is too small to analyze."}
    except Exception as exc:
        return {"landmark": "Unknown", "confidence": 0.0, "low_confidence": True, "message": f"Image could not be read: {exc}"}
    if not api_key:
        try:
            return _recognize_locally(image)
        except Exception as local_exc:
            return {"landmark": "Unknown image", "title": "Unknown image", "category": "unknown", "description": "The image was received but no local vision model was available.", "confidence": 0.0, "low_confidence": True, "message": f"No vision API key and local analysis failed: {local_exc}"}
    try:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_format = image.format.lower() if image.format else "jpeg"
        mime_type = "jpeg" if image_format == "jpg" else image_format
        response = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": model or "gpt-4o-mini", "temperature": 0, "max_tokens": 240, "messages": [{"role": "system", "content": "Analyze any uploaded image. If it contains a recognizable landmark, identify it; otherwise describe the main subject. Return only valid JSON with keys title, category, description, landmark, confidence. Confidence must be 0 to 1 and must be below 0.6 when exact identification is uncertain. Never invent an exact landmark name."}, {"role": "user", "content": [{"type": "text", "text": "Tell me what is in this photo, what kind of image it is, and any visible landmark or important details."}, {"type": "image_url", "image_url": {"url": f"data:image/{mime_type};base64,{encoded}"}}]}]}, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        result = json.loads(content)
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
        return {"landmark": result.get("landmark") or result.get("title", "Unknown landmark"), "title": result.get("title", "Image analysis"), "category": result.get("category", "image"), "description": result.get("description", "No description was returned."), "confidence": confidence, "low_confidence": confidence < 0.6, "message": "Analyzed with the vision model."}
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            try:
                return _recognize_locally(image)
            except Exception as local_exc:
                return {"landmark": "Unknown image", "title": "Unknown image", "category": "unknown", "description": "The image could not be described because both vision providers were unavailable.", "confidence": 0.0, "low_confidence": True, "message": f"OpenAI quota reached and local vision fallback was unavailable: {local_exc}."}
        return {"landmark": "Unknown landmark", "confidence": 0.0, "low_confidence": True, "message": f"Vision provider rejected the request ({exc.response.status_code if exc.response is not None else 'HTTP error'})."}
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
        return {"landmark": "Unknown landmark", "confidence": 0.0, "low_confidence": True, "message": f"Vision analysis failed safely: {exc}"}
