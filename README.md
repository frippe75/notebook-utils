# Notebook Utils
This repository contains utility functions for use within Jupyter notebooks. These utilities interact with various FaaS services, such as inpainting and segmentation.

## Features
- Inpainting Utility: Utilize FaaS services to perform image inpainting, filling in missing parts of an image based on a mask.
- Segmentation Utility: Leverage FaaS services to segment images, identifying and isolating different parts of an image.
- Environment Variable Configuration: Easy setup with environment variables for API keys and endpoint IDs.
- Modular Design: Organized into specific modules for inpainting and segmentation, allowing for easy extension and maintenance.
- Jupyter Notebook Integration: Designed to be used within Jupyter notebooks, making it ideal for data scientists and researchers.
- Example Usage: Provides example code snippets to help users get started quickly.

## Structure
- `cloud_service_utils/`: Contains specific implementations for inpainting and segmentation.

## Usage
To use the utilities, import the necessary functions from the respective modules:

```python
from cloud_service_utils.inpainting import inpaint_image_via_faas
from cloud_service_utils.segmentation import segment_image_via_faas
```

## Setup
Ensure you have the required environment variables set up:

- `RUNPOD_API_KEY`: Your API key for accessing the FaaS services.
- `RUNPOD_ENDPOINT_ID`: The endpoint ID for the specific service.

## Example

Here's an example of how to use the inpainting utility:

```python
import cv2
from cloud_service_utils.inpainting import inpaint_image_via_faas

image = cv2.imread("input.png")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

result = inpaint_image_via_faas(image=image, mask=mask, debug=True)
cv2.imwrite("output.png", result)
```
