import gradio as gr
import os
import pandas as pd
from src.application.services import BridgeModelService


class BridgeInference:
    """
    This class builds and launches the Gradio interface for the Bridge Model Inference.
    It uses the BridgeModelService directly to perform inference.
    """

    def __init__(self, checkpoint_path: str) -> None:
        self.service = BridgeModelService(checkpoint_path)

    def safe_generate_output(self, words: str, pathway: str) -> str:
        """
        Safely calls the domain service and returns output, wrapping errors for Gradio.
        """
        try:
            return self.service.generate_output(words, pathway)
        except ValueError as e:
            raise gr.Error(f"ValueError: {e}")
        except Exception as e:
            raise gr.Error(f"Unexpected Error: {e}")

    def get_phoneme_details(self):
        """
        Loads and returns the phoneme feature details from phonreps.csv.
        """
        csv_path = os.path.join('data', "phonreps.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
        else:
            raise gr.Error("phonreps.csv file not found.")

    def show_phoneme_details_callback(self):
        """
        Callback to show the phoneme details.
        Loads the CSV data and returns an update to the DataFrame component to make it visible,
        while toggling the visibility of the Show/Hide buttons.
        """
        df = self.get_phoneme_details()
        # Update the table to be visible and set its value to the CSV data.
        # Also, hide the Show button and reveal the Hide button.
        return gr.update(visible=True, value=df), gr.update(visible=False), gr.update(visible=True)

    def hide_phoneme_details_callback(self):
        """
        Callback to hide the phoneme details.
        Clears the DataFrame and toggles the visibility of the Show/Hide buttons.
        """
        # Hide the table and clear its value.
        return gr.update(visible=False, value=None), gr.update(visible=True), gr.update(visible=False)

    def build_ui(self) -> gr.Blocks:
        """
        Build and return the Gradio Blocks interface.
        """
        # Load custom CSS from the css folder.
        # To prevent text wrapping in the DataFrame, add the following CSS to your style.css:
        #
        #   .gradio-container table th,
        #   .gradio-container table td {
        #       white-space: nowrap;
        #   }
        #
        css_path = os.path.join(os.path.dirname(__file__), "css", "style.css")
        custom_css = ""
        if os.path.exists(css_path):
            with open(css_path, "r") as css_file:
                custom_css = css_file.read()

        with gr.Blocks(css=custom_css, title="Bridge Model Inference") as demo:
            # Header
            gr.Markdown("<h1 class='main-header'>🚀 Bridge Model Inference</h1>")

            # Description box (using HTML to avoid Markdown conflicts)
            with gr.Group():
                gr.HTML(
                    """
                    <div class="description-box">
                        <strong>Welcome!</strong><br><br>
                        This application takes a list of words and transforms them based on the selected pathway.
                        Explore orthographic or phonological transformations with ease.<br><br>
                        
                        <strong>How to use the app:</strong><br>
                        1. Enter words separated by spaces or commas (e.g., <code>cat, dog</code>).<br>
                        2. Select a transformation pathway.<br>
                        3. Click <strong>Submit</strong> to see the output.<br><br>
                        
                        <strong>Pathway Options:</strong><br>
                        - <strong>o2p</strong>: Orthographic → Phonological<br>
                        - <strong>p2o</strong>: Phonological → Orthographic<br>
                        - <strong>op2op</strong>: Orthographic-Phonological → Orthographic-Phonological<br>
                        - <strong>p2p</strong>: Phonological → Phonological<br>
                        - <strong>o2o</strong>: Orthographic → Orthographic
                    </div>
                    """
                )

            # Two-column layout for model input and output
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Input Words", placeholder="Enter words separated by spaces or commas"
                    )
                    pathway_radio = gr.Radio(
                        choices=["o2p", "p2o", "op2op", "p2p", "o2o"],
                        label="Select Transformation Pathway",
                        value="o2p",
                    )
                    submit_btn = gr.Button("Submit", variant="primary")
                with gr.Column():
                    output_text = gr.Textbox(
                        label="Generated Output", placeholder="Output will appear here...", lines=5
                    )

            # Bind the submit button action
            submit_btn.click(
                fn=lambda words, pathway: self.safe_generate_output(words, pathway),
                inputs=[input_text, pathway_radio],
                outputs=[output_text],
            )

            # Phoneme Details section (placed below the submit button)
            with gr.Row():
                phoneme_show_btn = gr.Button("Show Phoneme Details", visible=True)
                phoneme_hide_btn = gr.Button("Hide Phoneme Details", visible=False)
            phoneme_table = gr.DataFrame(label="Phoneme Feature Details", interactive=False, visible=False)

            # Set actions for the toggle buttons
            phoneme_show_btn.click(
                fn=self.show_phoneme_details_callback,
                inputs=[],  # No inputs needed
                outputs=[phoneme_table, phoneme_show_btn, phoneme_hide_btn],
            )

            phoneme_hide_btn.click(
                fn=self.hide_phoneme_details_callback,
                inputs=[],
                outputs=[phoneme_table, phoneme_show_btn, phoneme_hide_btn],
            )

        return demo

    def launch(self, share: bool = True) -> None:
        """
        Build and launch the Gradio UI.
        """
        demo = self.build_ui()
        demo.launch(share=share)
