# Tiny Déjà Vu: Smaller Memory Footprint & Faster Inference on Sensor Data Streams with Always-On Microcontrollers

This repository includes the core implementation parts of Tiny Déjà Vu.

## Reproduce Experimental Results on MCU

The experiments were conducted via [RIOT-ML](https://github.com/TinyPART/RIOT-ML/). To reproduce the results it is required to get RIOT-ML ready.

0. Go get a IoT board e.g. [STM32 nucleo-f767zi](https://www.st.com/en/evaluation-tools/nucleo-f767zi.html). Connect the IoT board to your PC.
1. Clone the RIOT-ML from https://github.com/TinyPART/RIOT-ML/.
2. Follow the [Prequisites](https://github.com/TinyPART/RIOT-ML/tree/main?tab=readme-ov-file#prequisites) section, install all necessary packages and toolchains.
3. Download the generated C-Code from [here](https://drive.google.com/file/d/1jUp2DE-8k0RMKWzXhI40D83JW7GyM1q8/view), and extract it to the to the RIOT-ML directory.
3. Copy `eval_ssm_HIL.py` and `main.c` to the RIOT-ML directory.
4. Run `python eval_ssm_HIL.py` under the RIOT-ML directory.
5. Grab some coffee and wait for the results written in `SSM_eval_result_{board}.json` :).

It is noted that `r` in this repo / results actually represents $1-r_{overlap}$.