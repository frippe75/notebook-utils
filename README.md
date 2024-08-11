# Notebook Utils
This repository contains utility functions for use within Jupyter notebooks. These utilities interact with various FaaS services, such as inpainting and segmentation.

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
