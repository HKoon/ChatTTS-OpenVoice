# import spaces
import os
import random
import argparse

import torch
import gradio as gr
import numpy as np

import ChatTTS

from OpenVoice.utils import se_extractor
from OpenVoice.api import ToneColorConverter
import soundfile

print("loading ChatTTS model...")
chat = ChatTTS.Chat()
chat.load_models()


def generate_seed():
    new_seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": new_seed
        }

# @spaces.GPU
def chat_tts(text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag, refine_text_input, output_path=None):

    torch.manual_seed(audio_seed_input)
    rand_spk = torch.randn(768)
    params_infer_code = {
        'spk_emb': rand_spk, 
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
        }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
    
    torch.manual_seed(text_seed_input)

    if refine_text_flag:
        if refine_text_input:
           params_refine_text['prompt'] = refine_text_input
        text = chat.infer(text, 
                          skip_refine_text=False,
                          refine_text_only=True,
                          params_refine_text=params_refine_text,
                          params_infer_code=params_infer_code
                          )
        print("Text has been refined!")
    
    wav = chat.infer(text, 
                     skip_refine_text=True, 
                     params_refine_text=params_refine_text, 
                     params_infer_code=params_infer_code
                     )
    
    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    text_data = text[0] if isinstance(text, list) else text

    if output_path is None:
        return [(sample_rate, audio_data), text_data]
    else:
        soundfile.write(output_path, audio_data, sample_rate)
        return text_data

# OpenVoice Clone
ckpt_converter = 'OpenVoice/checkpoints/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

def generate_audio(text, audio_ref, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag, refine_text_input):
    save_path = "output.wav"
    
    if audio_ref != "" :
      # Run the base speaker tts
      src_path = "tmp.wav"
      text_data = chat_tts(text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag, refine_text_input, src_path)
      print("Ready for voice cloning!")
    
      source_se, audio_name = se_extractor.get_se(src_path, tone_color_converter, target_dir='processed', vad=True)
      reference_speaker = audio_ref
      target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)

      print("Get voices segment!")
    
      # Run the tone color converter
      # convert from file
      tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path)
    else:
      chat_tts(text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag, refine_text_input, save_path)

    print("Finished!")

    return [save_path, text_data]


with gr.Blocks() as demo:
    gr.Markdown("# <center>🥳 ChatTTS x OpenVoice 🥳</center>")
    gr.Markdown("## <center>🌟 Make it sound super natural and switch it up to any voice you want, nailing the mood and tone also!🌟 </center>")

    default_text = "Today a man knocked on my door and asked for a small donation toward the local swimming pool. I gave him a glass of water."        
    text_input = gr.Textbox(label="Input Text", lines=4, placeholder="Please Input Text...", value=default_text)


    default_refine_text = "[oral_2][laugh_0][break_6]"    
    refine_text_checkbox = gr.Checkbox(label="Refine text", info="'oral' means add filler words, 'laugh' means add laughter, and 'break' means add a pause. (0-10) ", value=True)
    refine_text_input = gr.Textbox(label="Refine Prompt", lines=1, placeholder="Please Refine Prompt...", value=default_refine_text)
    with gr.Column():    
        voice_ref = gr.Audio(label="Reference Audio", type="filepath")

    with gr.Row():
        temperature_slider = gr.Slider(minimum=0.00001, maximum=1.0, step=0.00001, value=0.3, label="Audio temperature")
        top_p_slider = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.7, label="top_P")
        top_k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=20, label="top_K")

    with gr.Row():
        audio_seed_input = gr.Number(value=42, label="Speaker Seed")
        generate_audio_seed = gr.Button("\U0001F3B2")
        text_seed_input = gr.Number(value=42, label="Text Seed")
        generate_text_seed = gr.Button("\U0001F3B2")

    generate_button = gr.Button("Generate")
        
    text_output = gr.Textbox(label="Refined Text", interactive=False)
    audio_output = gr.Audio(label="Output Audio")

    generate_audio_seed.click(generate_seed, 
                              inputs=[], 
                              outputs=audio_seed_input)
        
    generate_text_seed.click(generate_seed, 
                             inputs=[], 
                             outputs=text_seed_input)
        
    generate_button.click(generate_audio, 
                          inputs=[text_input, voice_ref, temperature_slider, top_p_slider, top_k_slider, audio_seed_input, text_seed_input, refine_text_checkbox, refine_text_input], 
                          outputs=[audio_output,text_output])

parser = argparse.ArgumentParser(description='ChatTTS-OpenVoice Launch')
parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
parser.add_argument('--server_port', type=int, default=8080, help='Server port')
args = parser.parse_args()

# demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=True)

if __name__ == '__main__':
    demo.launch()
