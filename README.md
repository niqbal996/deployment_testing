# AI-based Maize and Weeds Detection on the Edge with CornWeed Dataset
This repository has python scripts to test the inference speed of object detector models in both ONNX and TensorRT on various edge device. 
Code for the paper:

[Naeem Iqbal](https://www.dfki.de/web/ueber-uns/mitarbeiter/person/naiq01), [Christoph Manss](https://www.dfki.de/web/ueber-uns/mitarbeiter/person/chma05),[Christian Scholz](https://www.hs-osnabrueck.de/forschung/recherche/laboreinrichtungen-und-versuchsbetriebe/labor-fuer-mikro-und-optoelektronik/team/#c605134), [Daniel Koenig](https://www.hs-osnabrueck.de/forschung/recherche/laboreinrichtungen-und-versuchsbetriebe/labor-fuer-mikro-und-optoelektronik/team/#c759389), [Matthias Igelbrink](https://www.hs-osnabrueck.de/forschung/recherche/laboreinrichtungen-und-versuchsbetriebe/labor-fuer-mikro-und-optoelektronik/team/#c605123), [Arno Ruckelshausen](	https://www.hs-osnabrueck.de/forschung/recherche/laboreinrichtungen-und-versuchsbetriebe/labor-fuer-mikro-und-optoelektronik/team/#c15056) 

"[AI-based Maize and Weeds Detection on the Edge with CornWeed Dataset](https://arxiv.org/abs/2105.08704)", FedCSIS AgriAI2023

## Container build
For `amd64` systems:
```bash
sudo -H DOCKER_BUILDKIT=1 docker build -f docker/amd64/Dockerfile -t <your-docker-repository>/ros_inference:noetic_onnx .
```
For `arm64` systems (tested on Jetson Orin AGX and Jetson Xavier NX):
```bash
sudo -H DOCKER_BUILDKIT=1 docker build -f docker/arm64/Dockerfile -t <your-docker-repository>/ros_inference:noetic_onnx .
```
## Running container
```bash
docker run -it --runtime=nvidia --net=host <your-docker-repository>/ros_inference:noetic_onnx 
```
## Acknowledgements
