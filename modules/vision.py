"""Lazy landmark recognition. A fine-tuned checkpoint can be plugged in via secrets/config."""
import base64
from io import BytesIO
import json
import re
from functools import lru_cache
from typing import Any
import requests
from PIL import ExifTags

LANDMARK_HINTS = {"hunza": "Hunza", "skardu": "Skardu", "lahore": "Lahore", "islamabad": "Islamabad"}
LOCAL_CANDIDATES = [
    "Minar-e-Pakistan Lahore", "Faisal Mosque Islamabad", "Badshahi Mosque Lahore", "Lahore Fort Pakistan",
    "Mazar-e-Quaid Karachi", "Pakistan Monument Islamabad", "Red Fort Delhi India", "Red Fort Lahore Pakistan",
    "Jama Masjid Delhi India", "Humayun's Tomb Delhi India",
    "Hunza Valley Pakistan", "Attabad Lake Pakistan",
    "Baltit Fort Pakistan", "Skardu Pakistan", "Deosai National Park Pakistan", "Malam Jabba Swat Pakistan",
    "Naran Kaghan Pakistan", "Chitral Pakistan", "Kashmir Pakistan", "Swat Valley Pakistan", "a natural mountain landscape",
    "a turquoise alpine lake", "a green valley with a river", "a high altitude desert landscape", "a forest waterfall",
    "a city street", "a mosque or religious building",
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
    "Hunza Valley Pakistan": "Hunza Valley in northern Pakistan, with sharp rocky peaks, terraced green slopes, dry mountain valleys, and clear alpine light",
    "Attabad Lake Pakistan": "Attabad Lake in Hunza, a bright turquoise mountain lake surrounded by steep bare rock peaks",
    "Skardu Pakistan": "Skardu in Gilgit-Baltistan, with broad dry mountain valleys, rocky peaks, blue skies, and high-altitude terrain",
    "Deosai National Park Pakistan": "Deosai National Park, a wide high-altitude plateau with open grassland, rounded mountains, and minimal trees",
    "Malam Jabba Swat Pakistan": "Malam Jabba in Swat, a green forested mountain resort with rolling hills and cool highland scenery",
    "Naran Kaghan Pakistan": "Naran Kaghan Valley, a green Himalayan valley with a river, pine forests, and steep mountain slopes",
    "Kashmir Pakistan": "Kashmir, a lush mountain region with green valleys, rivers, forests, and layered Himalayan hills",
    "Swat Valley Pakistan": "Swat Valley, a green valley with rivers, forests, farmland, and surrounding mountains",
}


def _gps_from_image(image: Any) -> dict[str, Any] | None:
    """Read embedded GPS metadata when a camera or phone included it."""
    try:
        exif = image.getexif()
        gps_info = exif.get(34853)
        if not gps_info:
            return None
        gps = {ExifTags.GPSTAGS.get(key, key): value for key, value in gps_info.items()}
        coordinates = gps.get("GPSLatitude"), gps.get("GPSLongitude")
        if not all(coordinates) or not gps.get("GPSLatitudeRef") or not gps.get("GPSLongitudeRef"):
            return None

        def decimal(value: tuple[Any, Any, Any], reference: str) -> float:
            degrees, minutes, seconds = [float(part) for part in value]
            result = degrees + minutes / 60 + seconds / 3600
            return -result if reference in ("S", "W") else result

        latitude = decimal(coordinates[0], gps["GPSLatitudeRef"])
        longitude = decimal(coordinates[1], gps["GPSLongitudeRef"])
        return {"latitude": round(latitude, 6), "longitude": round(longitude, 6), "map_url": f"https://www.google.com/maps?q={latitude},{longitude}"}
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


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
        metadata_location = _gps_from_image(image)
    except Exception as exc:
        return {"landmark": "Unknown", "confidence": 0.0, "low_confidence": True, "message": f"Image could not be read: {exc}"}
    if not api_key:
        try:
            result = _recognize_locally(image)
            result["metadata_location"] = metadata_location
            return result
        except Exception as local_exc:
            return {"landmark": "Unknown image", "title": "Unknown image", "category": "unknown", "description": "The image was received but no local vision model was available.", "confidence": 0.0, "low_confidence": True, "message": f"No vision API key and local analysis failed: {local_exc}"}
    try:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_format = image.format.lower() if image.format else "jpeg"
        mime_type = "jpeg" if image_format == "jpg" else image_format
        response = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": model or "gpt-4o-mini", "temperature": 0, "max_tokens": 300, "messages": [{"role": "system", "content": "Analyze any uploaded image. If it contains a recognizable landmark or nature scene, identify the most likely place. For an ordinary landscape, give a possible region or destination only when visual evidence supports it, and explain the clues. Return only valid JSON with keys title, category, description, landmark, confidence. Confidence must be 0 to 1 and must be below 0.6 when exact identification is uncertain. Never invent an exact landmark name or claim certainty from scenery alone."}, {"role": "user", "content": [{"type": "text", "text": "Tell me what is in this photo, what kind of image it is, the most likely place or possible region if supported, and the visual clues. For nature photos consider Hunza, Skardu, Swat, Naran Kaghan, Kashmir, and other likely regions, but say unknown when the image cannot establish a location."}, {"type": "image_url", "image_url": {"url": f"data:image/{mime_type};base64,{encoded}"}}]}]}, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        result = json.loads(content)
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
        return {"landmark": result.get("landmark") or result.get("title", "Unknown landmark"), "title": result.get("title", "Image analysis"), "category": result.get("category", "image"), "description": result.get("description", "No description was returned."), "confidence": confidence, "low_confidence": confidence < 0.6, "message": "Analyzed with the vision model.", "metadata_location": metadata_location}
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            try:
                return _recognize_locally(image)
            except Exception as local_exc:
                return {"landmark": "Unknown image", "title": "Unknown image", "category": "unknown", "description": "The image could not be described because both vision providers were unavailable.", "confidence": 0.0, "low_confidence": True, "message": f"OpenAI quota reached and local vision fallback was unavailable: {local_exc}."}
        return {"landmark": "Unknown landmark", "confidence": 0.0, "low_confidence": True, "message": f"Vision provider rejected the request ({exc.response.status_code if exc.response is not None else 'HTTP error'})."}
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
        return {"landmark": "Unknown landmark", "confidence": 0.0, "low_confidence": True, "message": f"Vision analysis failed safely: {exc}"}


def describe_image(image_bytes: bytes, api_key: str | None = None, model: str | None = None) -> dict[str, str]:
    """Return a practical travel-focused description of everything visible in an image."""
    def offline_description(message: str) -> dict[str, str]:
        try:
            result = recognize_landmark(image_bytes, None, None)
            category = result.get("category", "image")
            label = result.get("landmark", "an unknown image")
            return {"summary": f"Offline visual match: {label}.", "objects": f"The offline model classified this as {category}.", "food_or_items": "Not available in offline mode.", "setting": result.get("description", "Broad visual category only."), "visible_text": "Text reading is not available in offline mode.", "travel_context": message}
        except Exception:
            return {"description": "Detailed image analysis is temporarily unavailable.", "message": message}

    if not api_key:
        return offline_description("Configure OPENAI_API_KEY for object-level image details and text reading.")
    try:
        from PIL import Image
        image = Image.open(BytesIO(image_bytes))
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_format = image.format.lower() if image.format else "jpeg"
        mime_type = "jpeg" if image_format == "jpg" else image_format
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model or "gpt-4o-mini", "temperature": 0, "max_tokens": 450, "messages": [
                {"role": "system", "content": "Describe the uploaded travel image accurately. Return valid JSON with keys: summary, objects, food_or_items, setting, visible_text, travel_context. Use short strings, do not invent details, and say 'Not visible' when uncertain."},
                {"role": "user", "content": [{"type": "text", "text": "What exactly is visible in this photo? List the main objects or food, describe the setting, transcribe readable signs or menu text, and give useful travel context."}, {"type": "image_url", "image_url": {"url": f"data:image/{mime_type};base64,{encoded}"}}]},
            ]},
            timeout=30,
        )
        response.raise_for_status()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.json()["choices"][0]["message"]["content"].strip(), flags=re.IGNORECASE)
        result = json.loads(content)
        return {key: str(result.get(key) or "Not visible") for key in ("summary", "objects", "food_or_items", "setting", "visible_text", "travel_context")}
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            return offline_description("OpenAI rate limit or quota reached. Showing the offline visual fallback; check billing, usage limits, or wait before trying again.")
        return offline_description(f"OpenAI rejected the image request ({exc.response.status_code if exc.response is not None else 'HTTP error'}).")
    except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError, OSError) as exc:
        return offline_description(f"Detailed image service unavailable. Showing offline visual fallback: {exc}")
