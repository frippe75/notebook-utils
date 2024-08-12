import time
import requests
import base64
import os
import cv2
import numpy as np

# Hard coded pod id
RUNPOD_ENDPOINT_ID = "906cbtg1541h5c"

def inpaint_image_via_faas(image=None, image_path=None, mask=None, mask_path=None, output_path="result.png", endpoint_id=RUNPOD_ENDPOINT_ID, debug=False):
    """
    Sends a request to the inpainting FaaS service and processes the response.

    Parameters:
    - image: cv2 image object, the input image (optional if image_path is provided)
    - image_path: str, path to the input image file (optional if image is provided)
    - mask: cv2 image object, the mask image (optional if mask_path is provided)
    - mask_path: str, path to the mask image file (optional if mask is provided)
    - output_path: str, path to save the inpainted result image (only used if image_path is provided)
    - endpoint_id: str, the ID of the FaaS endpoint (default is the constant RUNPOD_ENDPOINT_ID)
    - debug: bool, if True, prints debug information

    Returns:
    - If image or mask are provided via cv2, returns a cv2 image object.
    - If image_path and mask_path are provided, saves the result to output_path and returns None.
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

    # Convert mask to base64
    if mask is not None:
        _, buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
    elif mask_path is not None:
        with open(mask_path, "rb") as mask_file:
            mask_base64 = base64.b64encode(mask_file.read()).decode('utf-8')
    else:
        raise ValueError("Either mask or mask_path must be provided.")

    # Create JSON payload for the inpainting FaaS
    payload = {
        "input": {
            "image": image_base64,
            "mask": mask_base64
        }
    }

    if debug:
        print(f"POST {endpoint_url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

    # Start timing the entire request process
    frontend_start_time = time.time()

    # Send the POST request
    response = requests.post(endpoint_url, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }, data=json.dumps(payload))

    # Measure the total time taken for the request from the frontend perspective
    frontend_total_time = time.time() - frontend_start_time

    if debug:
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")

    # Check if the response is successful
    response.raise_for_status()

    # Parse the response
    response_json = response.json()
    output_image_base64 = response_json.get("output_image", "")
    stats = response_json.get("stats", {})

    if not output_image_base64:
        raise ValueError("No output image in the response.")

    # Decode the output image from base64
    output_image_data = base64.b64decode(output_image_base64)

    # Display timing information
    if debug:
        print(f"Inference Time (Backend): {stats.get('inference_time', 'N/A')} seconds")
        print(f"Overall Time (Backend): {stats.get('overall_time', 'N/A')} seconds")
        print(f"Total Time (Frontend): {frontend_total_time:.2f} seconds")

    if image is not None or mask is not None:
        # Convert the output image data back to a cv2 image object
        nparr = np.frombuffer(output_image_data, np.uint8)
        result_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return result_image
    else:
        # Save the output image to a file
        with open(output_path, "wb") as out_file:
            out_file.write(output_image_data)
        print(f"Inpainted image saved as {output_path}")
        return None
