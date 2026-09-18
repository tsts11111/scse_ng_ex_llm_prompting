## Import the necessary modules
import json
import ollama

## Import the function from the module parse_data
from parse_data import load_items, get_unclaimed_items, save_result


## Build your prompt based on the description the user provides
## and the items that are available in the lost-and-found database.
## The model must follow the rules listed in the README file
## The function should return the system prompt and the user prompt.
## You may need to use json.dumps() to convert the available_items list into a JSON string.
def build_prompt(description, available_items):
    system_prompt = (
        "You are a campus lost-and-found assistant. Your job is to match a lost "
        "item described by the user against the items in a lost-and-found database.\n"
        "Rules:\n"
        "- Use ONLY the items from the provided JSON database. Do not invent items.\n"
        "- Not all details of an item must match to be a possible match.\n"
        "- Return ONLY valid JSON with EXACTLY this structure:\n"
        '{"matches": ["ITEM_ID"], "confidence": "LOW"}\n'
        '- "matches" must contain all possible matches as item IDs (empty list if none).\n'
        '- "confidence" must be exactly one of: LOW, MEDIUM, HIGH.\n'
        "- Do not include any other text, explanation, or markdown formatting."
    )
    user_prompt = (
        "Lost item description: {}\n\n"
        "Available items in the lost-and-found database:\n{}".format(
            description, json.dumps(available_items, ensure_ascii=False)
        )
    )
    return system_prompt, user_prompt


## Logic to ask Qwen for all the possible matches based on the system prompt and user prompt.
## The function should return the response from Qwen.
def ask_qwen(system_prompt, user_prompt):
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


## Logic to parse the response from Qwen and return the result.
## You may need to use json.loads() to convert the response string into a suitable Python data structure.
def parse_response(response_text):
    text = response_text.strip()
    # Strip markdown code fences if the model wrapped the JSON in them.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to extracting the first {...} block if extra text was added.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


## Logic to validate the result returned by Qwen.
## It should check if the result is a dictionary, contains the keys "matches" and "confidence", and that the values are of the correct type.
## If everything is correct, then it should check if the item IDs in the "matches" list are valid IDs .
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False
    if "matches" not in result or "confidence" not in result:
        return False
    matches = result["matches"]
    confidence = result["confidence"]
    if not isinstance(matches, list) or not isinstance(confidence, str):
        return False
    if confidence not in ("LOW", "MEDIUM", "HIGH"):
        return False
    if not all(isinstance(item_id, str) for item_id in matches):
        return False
    valid_ids = {item["id"] for item in available_items}
    if not all(item_id in valid_ids for item_id in matches):
        return False
    return True


## Logic to display the matches found by Qwen in a user-friendly format.
## It should look something like this:
"""
CAMPUS LOST-AND-FOUND ASSISTANT
==================================================

Describe the item you lost: I lost a black bag somewhere

Searching for possible matches...

MATCH RESULT
--------------------------------------------------
Confidence: MEDIUM

Possible matches:

ID: F101
Item: backpack
Color: black
Location: Library 2nd floor
Date found: 2026-09-15

Result saved to output/match_result.json
 """
## If no matches are found, it should display a message indicating that no matches were found, along with the empty list
def display_matches(result, available_items):
    items_by_id = {item["id"]: item for item in available_items}
    matches = result["matches"]
    confidence = result["confidence"]

    print()
    print("MATCH RESULT")
    print("--------------------------------------------------")
    print("Confidence: {}".format(confidence))
    print()

    if not matches:
        print("No matches were found for your item.")
        print("Possible matches: []")
        return

    print("Possible matches:")
    print()
    for item_id in matches:
        item = items_by_id[item_id]
        print("ID: {}".format(item["id"]))
        print("Item: {}".format(item["item"]))
        print("Color: {}".format(item["color"]))
        print("Location: {}".format(item["location"]))
        print("Date found: {}".format(item["date"]))
        print()


## Control center for the entire program.
def main():
    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("==================================================")
    print()

    items = load_items("found_items.json")
    available_items = get_unclaimed_items(items)

    description = input("Describe the item you lost: ").strip()

    print()
    print("Searching for possible matches...")

    system_prompt, user_prompt = build_prompt(description, available_items)
    response_text = ask_qwen(system_prompt, user_prompt)
    result = parse_response(response_text)

    if not validate_result(result, available_items):
        print("Warning: the model returned an invalid result. Showing raw response:")
        print(response_text)
        result = {"matches": [], "confidence": "LOW"}

    display_matches(result, available_items)

    save_result(result, "output/match_result.json")
    print("Result saved to output/match_result.json")


if __name__ == "__main__":
    main()
