<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<!-- <div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</div> -->



## Fashion Style Transfer: A Bidirectional Fusion Approach with Commonality Style-enhanced and Content-style Features Decoupled



## Getting Started

### Prerequisites

For packages, see environment.yaml.

  ```sh
  conda env create -f environment.yaml
  conda activate fashionstyle
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Installation

   Clone the repo
   ```sh
   git clone https://github.com/Phoenixjk/fashionstyle.git
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Datasets
   Place fashion items, content images, and style images into three separate folders.
   Example directory hierarchy:
   ```sh
      fashionstyle
      |--- Images
             |--- fuzhuang
      |--- datasets
             |--- edges2shoes
                   |--- fengge
                   |--- neirong
                   
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Train

   Train:
   ```sh
   python train3.py
   ```
   See `configs/stable-diffusion/v1-finetune.yaml` for more options
   
   Download the pretrained [Stable Diffusion Model](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt) and save it at ./models/sd/sd-v1-4.ckpt.
   <p align="right">(<a href="#top">back to top</a>)</p>
   


