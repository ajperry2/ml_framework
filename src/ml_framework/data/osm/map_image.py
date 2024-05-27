"""A module which lets you plot images of the earth styles I publish.

All data is provided by Mapbox:
https://www.mapbox.com/

Available styles currently include:
- satellite
- street

Example Usage:

tile = get_mapbox_image(
    min_longitude=45.720-0.04/2, min_latitude=4.210-0.08/2,
    max_longitude=45.720+0.04/2, max_latitude=4.210+0.08/2,
    style_name="satellite",
    image_width=512, image_height=256
)

"""
import os
from PIL import Image
from io import BytesIO
import urllib3

style_name_to_id = {
    "satellite": "clwnz8d1c00vx01po704o3mqc",
    "street": "clwo1jvgr00qx01pp8dbl2zlz"
}


def get_mapbox_image(
    min_longitude: float, min_latitude: float,
    max_longitude: float, max_latitude: float,
    style_name="satellite",
    image_width: int = 256, image_height: int = 256
) -> Image:
    """Download an image from Mapbox using a custom style.

    Arguments:
        min_longitude (float):
            Minimum longitude of the images bounding box.
        min_latitude (float):
            Minimum latitude of the images bounding box.
        max_longitude (float):
            Maximum longitude of the images bounding box.
        max_latitude (float):
            Maximum latitude of the images bounding box.
        style_name (str):
            Name of the Maps style.
            Should be in "satellite" | "street"
        image_width (int):
            Image width in pixels
        image_height (str):
            Image height in pixels

    Attribution:
    All data provided by this function was
    created by mapbox
    © Mapbox: https://www.mapbox.com/about/maps/

    More information can be found in the README in the same
    Folder as this code
    """
    user = os.environ["MAPBOX_USERNAME"]
    token = os.environ["MAPBOX_ACCESS_TOKEN"]
    assert style_name in style_name_to_id, \
        f"Style Name Must be in : {style_name_to_id.keys()}"
    style_id = style_name_to_id[style_name]
    bounding_box = f"[{min_longitude},{min_latitude}," +\
        f"{max_longitude},{max_latitude}]"
    # See this codes readme for attribution to mapbox and OSM
    image_url =  \
        f"https://api.mapbox.com/styles/v1/{user}/{style_id}/" +\
        f"static/{bounding_box}/{image_width}x{image_height}" +\
        f"?access_token={token}" +\
        "&logo=false" +\
        "&attribution=false"
    r = urllib3.request("GET", image_url)
    tile = Image.open(BytesIO(r.data))
    return tile