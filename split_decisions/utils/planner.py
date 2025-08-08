import os, re, json, ast, html, cv2
import numpy as np
from PIL import Image
import google.generativeai as genai

# ── Gemini setup ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ── helpers ───────────────────────────────────────────────────────────────────
def extract_json(raw: str):
    """
    Return the first JSON array inside `raw`.
    Returns None when nothing parseable is found.
    """
    # unwrap ```json fences
    fence = re.match(r"\s*```(?:json)?\s*([\s\S]*?)\s*```", raw)
    content = fence.group(1) if fence else raw

    m = re.search(r"\[[\s\S]*\]", content)          # first [...] block
    if not m:
        return None
    j_str = html.unescape(m.group(0))
    try:
        return json.loads(j_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(j_str)
        except Exception:
            return None


def _get_text_from_response(resp) -> str:
    """Concatenate *all* text parts from the first candidate."""
    texts = []
    for part in resp.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            texts.append(part.text)
    return "\n".join(texts)


# ── main API ──────────────────────────────────────────────────────────────────
def plan_task(task_description,
              image,
              model_name="gemini-2.5-pro-preview-05-06",
              temperature=0.0,
              max_output_tokens=30000,
              image_size=800,
              visualize=True,
              verbose=True,
              max_retries=1,
              idx=0):

    print("[INFO]: Generating Costmap...")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize((image_size, image_size))

    with open("split_decisions/prompts/planner.txt", encoding="utf-8") as f:
        prompt = f.read().replace("TASK_DESCRIPTION", task_description)

    model = genai.GenerativeModel(model_name)

    traj = None
    for attempt in range(max_retries + 1):
        resp = model.generate_content(
            [image, prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

        raw = _get_text_from_response(resp)

        if verbose:
            print("=" * 60, f"\n[Gemini raw] attempt {attempt}\n", raw[:1500],
                  "\n" + "=" * 60)
            if not raw.strip():               # no text at all → show protobuf
                print("[Gemini protobuf dump]", resp)

        traj = extract_json(raw)
        if traj is not None:
            break

    if traj is None:
        print("[Planner] No valid JSON after retries")
        return None

    # --- visualisation unchanged ---------------------------------------------
    if visualize:
        vis = np.array(image)
        for pt in traj:
            y, x = pt["point"]
            y = int(y / 1000 * image_size)
            x = int(x / 1000 * image_size)
            cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(vis, str(pt["label"]), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        Image.fromarray(vis).save(f"planner_output_{idx}.png")

    return traj
