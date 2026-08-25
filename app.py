from __future__ import annotations

import inspect
from typing import Any

import gradio as gr

from roommorph.config import (
    MAX_CFG_SCALE,
    MAX_SEED,
    MAX_STEPS,
    MAX_STRENGTH,
    MIN_CFG_SCALE,
    MIN_STEPS,
    MIN_STRENGTH,
    RUN_DEVICE,
)
from roommorph.generation import generate_design, generate_new_room_concept
from roommorph.model_manager import initial_status as _initial_status
from roommorph.prompts import NEGATIVE_PROMPT, STYLE_PROMPTS


def update_cfg_scale_interactivity(enabled: bool) -> dict[str, Any]:
    return gr.update(interactive=bool(enabled))


APP_CSS = """
.gradio-container {
    max-width: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
#app-shell {
    box-sizing: border-box;
    max-width: 1180px !important;
    width: calc(100% - 32px) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
#app-header, #app-header * { text-align: center; }
#runtime-status { padding: 10px 12px; border-left: 3px solid var(--primary-500); }
#settings-panel { width: 100%; }
#generate-button { min-height: 44px; }
#main-tabs {
    width: 100%;
    min-width: 0 !important;
    max-width: 100%;
}
#main-tabs > .tab-wrapper {
    width: 100%;
    min-width: 0;
    max-width: 100%;
    height: auto;
    min-height: 44px;
}
#main-tabs > .tab-wrapper > .tab-container[role="tablist"] {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0 !important;
    width: 100%;
    min-width: 0;
    max-width: 100%;
    height: auto;
    min-height: 44px;
}
#main-tabs > .tab-wrapper > .tab-container[role="tablist"] > button[role="tab"] {
    box-sizing: border-box;
    width: 100% !important;
    min-width: 0 !important;
    height: auto;
    min-height: 44px;
    padding: 8px 12px;
    justify-content: center;
    text-align: center;
    white-space: normal;
    line-height: 1.25;
}
#main-tabs > .tab-wrapper > .tab-container.visually-hidden > button {
    width: 50% !important;
    min-width: 0 !important;
    max-width: 50% !important;
    padding: 0 !important;
    overflow: hidden;
}
#main-tabs > .tab-wrapper > .overflow-menu { display: none !important; }
@media (max-width: 480px) {
    #main-tabs > .tab-wrapper > .tab-container[role="tablist"] > button[role="tab"] {
        padding: 8px 6px;
    }
}
"""

GRADIO_LAUNCH_ACCEPTS_CSS = "css" in inspect.signature(gr.Blocks.launch).parameters


def create_demo() -> gr.Blocks:
    blocks_kwargs: dict[str, Any] = {
        "title": "RoomMorph AI",
        "analytics_enabled": False,
    }
    if not GRADIO_LAUNCH_ACCEPTS_CSS:
        blocks_kwargs["css"] = APP_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Column(elem_id="app-shell", min_width=0):
            with gr.Column(elem_id="app-header", min_width=0):
                gr.Markdown(
                    "# RoomMorph AI\n"
                    "Redesign an existing room or generate a new interior concept using "
                    "this repository's custom Stable Diffusion 1.5 pipeline."
                )
                gr.Markdown(
                    f"**Runtime device:** `{RUN_DEVICE.type}`  \n"
                    "Free CPU inference can take several minutes. MPS or CUDA is recommended "
                    "for local use."
                )
            gr.Markdown(_initial_status(), elem_id="runtime-status")

            with gr.Tabs(elem_id="main-tabs"):
                with gr.Tab("Redesign Existing Room"):
                    with gr.Row(equal_height=True):
                        input_image = gr.Image(
                            label="Original room",
                            sources=["upload"],
                            type="pil",
                            image_mode=None,
                            format="png",
                            height=420,
                        )
                        generated_image = gr.Image(
                            label="Generated design",
                            type="pil",
                            image_mode="RGB",
                            format="png",
                            interactive=False,
                            height=420,
                        )

                    with gr.Group(elem_id="settings-panel"):
                        gr.Markdown("### Generation settings")

                        with gr.Row(equal_height=True):
                            style = gr.Dropdown(
                                choices=list(STYLE_PROMPTS),
                                value="Japandi",
                                label="Interior style",
                            )
                            resolution = gr.Radio(
                                choices=[("256 x 256", 256), ("512 x 512", 512)],
                                value=256,
                                label="Resolution",
                            )

                        instruction = gr.Textbox(
                            label="Additional design instruction",
                            placeholder="Example: keep the windows and add more storage",
                            lines=3,
                            max_lines=5,
                        )

                        with gr.Row(equal_height=True):
                            strength = gr.Number(
                                value=0.60,
                                minimum=MIN_STRENGTH,
                                maximum=MAX_STRENGTH,
                                step=0.05,
                                precision=2,
                                label="Strength",
                                info="How far the design can move from the original room.",
                            )
                            steps = gr.Number(
                                value=10,
                                minimum=MIN_STEPS,
                                maximum=MAX_STEPS,
                                step=1,
                                precision=0,
                                label="Inference steps",
                                info="More steps can add detail but take longer.",
                            )

                        with gr.Row(equal_height=True):
                            do_cfg = gr.Checkbox(
                                value=False,
                                label="Enable classifier-free guidance (CFG)",
                                info="Usually improves prompt adherence but increases computation.",
                            )
                            cfg_scale = gr.Number(
                                value=7.5,
                                minimum=MIN_CFG_SCALE,
                                maximum=MAX_CFG_SCALE,
                                step=0.5,
                                precision=1,
                                label="CFG scale (used only when CFG is enabled)",
                                info="Prompt guidance strength; available only when CFG is enabled.",
                                interactive=False,
                            )
                            seed = gr.Number(
                                value=42,
                                minimum=0,
                                maximum=MAX_SEED,
                                step=1,
                                precision=0,
                                label="Seed",
                                info="Repeat a result with the same settings.",
                            )

                    generate_button = gr.Button(
                        "Generate Design", variant="primary", elem_id="generate-button"
                    )
                    generation_status = gr.Textbox(
                        value=_initial_status(),
                        label="Generation status and metadata",
                        lines=11,
                        interactive=False,
                    )

                    do_cfg.change(
                        fn=update_cfg_scale_interactivity,
                        inputs=do_cfg,
                        outputs=cfg_scale,
                        queue=False,
                        show_progress="hidden",
                    )

                    generate_button.click(
                        fn=generate_design,
                        inputs=[
                            input_image,
                            style,
                            instruction,
                            strength,
                            steps,
                            do_cfg,
                            cfg_scale,
                            seed,
                            resolution,
                        ],
                        outputs=[generated_image, generation_status],
                        concurrency_limit=1,
                        concurrency_id="roommorph-inference",
                        show_progress="full",
                    )

                with gr.Tab("Generate New Room Concept"):
                    with gr.Row(equal_height=True):
                        with gr.Column(min_width=280):
                            txt2img_style = gr.Dropdown(
                                choices=list(STYLE_PROMPTS),
                                value="Japandi",
                                label="Room style preset",
                            )
                            txt2img_positive_prompt = gr.Textbox(
                                label="Positive prompt",
                                placeholder="Example: a sunlit open-plan living room with built-in shelves",
                                lines=5,
                                max_lines=7,
                            )
                            txt2img_negative_prompt = gr.Textbox(
                                value=NEGATIVE_PROMPT,
                                label="Negative prompt",
                                lines=4,
                                max_lines=6,
                            )

                        with gr.Column(min_width=280):
                            txt2img_generated_image = gr.Image(
                                label="Generated room concept",
                                type="pil",
                                image_mode="RGB",
                                format="png",
                                interactive=False,
                                height=420,
                            )

                    with gr.Group(elem_id="txt2img-settings-panel"):
                        gr.Markdown("### Generation settings")
                        with gr.Row(equal_height=True):
                            txt2img_resolution = gr.Radio(
                                choices=[("256 x 256", 256), ("512 x 512", 512)],
                                value=256,
                                label="Resolution",
                            )
                            txt2img_steps = gr.Number(
                                value=10,
                                minimum=MIN_STEPS,
                                maximum=MAX_STEPS,
                                step=1,
                                precision=0,
                                label="Inference steps",
                                info="More steps can add detail but take longer.",
                            )
                            txt2img_cfg_scale = gr.Number(
                                value=7.5,
                                minimum=MIN_CFG_SCALE,
                                maximum=MAX_CFG_SCALE,
                                step=0.5,
                                precision=1,
                                label="CFG scale",
                                info="Higher values follow the prompt more strongly.",
                            )
                            txt2img_seed = gr.Number(
                                value=42,
                                minimum=0,
                                maximum=MAX_SEED,
                                step=1,
                                precision=0,
                                label="Seed",
                                info="Repeat a result with the same settings.",
                            )

                    txt2img_generate_button = gr.Button(
                        "Generate Room Concept",
                        variant="primary",
                        elem_id="txt2img-generate-button",
                    )
                    txt2img_generation_status = gr.Textbox(
                        value=_initial_status(),
                        label="Text-to-image generation metadata",
                        lines=14,
                        interactive=False,
                    )

                    txt2img_generate_button.click(
                        fn=generate_new_room_concept,
                        inputs=[
                            txt2img_style,
                            txt2img_positive_prompt,
                            txt2img_negative_prompt,
                            txt2img_resolution,
                            txt2img_steps,
                            txt2img_cfg_scale,
                            txt2img_seed,
                        ],
                        outputs=[
                            txt2img_generated_image,
                            txt2img_generation_status,
                        ],
                        concurrency_limit=1,
                        concurrency_id="roommorph-inference",
                        show_progress="full",
                    )

    return demo.queue(max_size=8, default_concurrency_limit=1, api_open=False)


demo = create_demo()


if __name__ == "__main__":
    launch_kwargs = {"css": APP_CSS} if GRADIO_LAUNCH_ACCEPTS_CSS else {}
    demo.launch(**launch_kwargs)
