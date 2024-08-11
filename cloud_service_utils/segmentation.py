# segmentation.py
# This file contains the functions to interact with the segmentation FaaS service

import requests
import base64
import os
import cv2
import numpy as np

RUNPOD_ENDPOINT_ID = "your_default_segmentation_endpoint_id_here"

def segment_image_via_faas(image=None, image_path=None, class_names=None, output_path="result.json", endpoint_id=RUNPOD_ENDPOINT_ID, debug=False):
    """
    Sends a request to the segmentation FaaS service and processes the response.

    Parameters:
    - image: cv2 image object, the input image (optional if image_path is provided)
    - image_path: str, path to the input image file (optional if image is provided)
    - class_names: list of str, specific class names to segment (optional)
    - output_path: str, path to save the segmentation result (only used if image_path is provided)
    - endpoint_id: str, the ID of the FaaS endpoint (default is the constant RUNPOD_ENDPOINT_ID)
    - debug: bool, if True, prints debug information

    Returns:
    - If image is provided via cv2, returns a dict containing 'masks' and 'bounding_boxes'.
    - If image_path is provided, saves the result to output_path and returns None.
    """
    
    # Load API key from environment variable
    api_key = os.getenv('RUNPOD_API_KEY')
    if not api_key:
        raise ValueError("RUNPOD_API_KEY environment variable is not set.")

    # Create the full endpoint URL
    endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

    # Convert image to base64
    if image is not None:
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
    elif image_path is not None:
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    else:
        raise ValueError("Either image or image_path must be provided.")

    # Create JSON payload for the segmentation FaaS
    payload = {
        "input": {
            "image": image_base64,
            "class_names": class_names
        }
    }

    if debug:
        print(f"POST {endpoint_url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

    # Send the POST request
    response = requests.post(endpoint_url, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }, data=json.dumps(payload))

    if debug:
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")

    # Check if the response is successful
    response.raise_for_status()

    # Parse the response
    response_json = response.json()
    
    # Process the response and return the results if cv2 image was provided
    if image is not None:
        return {
            "masks": [cv2.imdecode(np.frombuffer(base64.b64decode(mask), np.uint8), cv2.IMREAD_COLOR) for mask in response_json.get('masks', [])],
            "bounding_boxes": response_json.get('bounding_boxes', [])
        }
    else:
        # Save the segmentation result to a file
        with open(output_path, "w") as out_file:
            json.dump(response_json, out_file)
        print(f"Segmentation result saved as {output_path}")
        return None
