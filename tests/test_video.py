from PIL import Image

from omlx.api.openai_models import ContentPart
from omlx.utils.image import extract_images_from_messages
from omlx.utils.video import _sample_indices, expand_video_parts


def test_sample_indices_are_bounded_and_chronological():
    assert _sample_indices(100, 4) == [0, 33, 66, 99]
    assert _sample_indices(3, 16) == [0, 1, 2]
    assert _sample_indices(0, 16) == []


def test_content_part_preserves_video_url():
    part = ContentPart(
        type="video_url",
        video_url={"url": "data:video/mp4;base64,AAAA"},
    )

    assert part.video_url is not None
    assert part.video_url.url.startswith("data:video/mp4")


def test_video_parts_expand_to_images_in_original_order(monkeypatch):
    first = Image.new("RGB", (2, 2), "red")
    second = Image.new("RGB", (2, 2), "blue")
    monkeypatch.setattr(
        "omlx.utils.video._decode_video_frames",
        lambda _url, _max_frames: [first, second],
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {
                    "type": "video_url",
                    "video_url": {"url": "data:video/mp4;base64,AAAA"},
                },
                {"type": "text", "text": "after"},
            ],
        }
    ]

    expanded = expand_video_parts(messages)
    text_messages, images, audio = extract_images_from_messages(expanded)

    assert [part["type"] for part in expanded[0]["content"]] == [
        "text",
        "image",
        "image",
        "text",
    ]
    assert text_messages == [{"role": "user", "content": "before\nafter"}]
    assert [image.getpixel((0, 0)) for image in images] == [
        (255, 0, 0),
        (0, 0, 255),
    ]
    assert audio == []
