# Unifying Short and Long-Term Tracking with Graph Hierarchies :sushi:

Official repository of our **CVPR 2023** paper 

> [**Unifying Short and Long-Term Tracking with Graph Hierarchies**](https://arxiv.org/abs/2212.03038)
> 
> [Orcun Cetintas*](https://dvl.in.tum.de/team/cetintas/), [Guillem Brasó*](https://dvl.in.tum.de/team/braso/), [Laura Leal-Taixe](https://dvl.in.tum.de/team/lealtaixe/)
> 
## Abstract
Tracking  objects over long videos effectively means solving a spectrum of problems, from short-term association for un-occluded objects to long-term association for objects that are occluded and then reappear in the scene. Methods tackling these two tasks are often disjoint and crafted for specific scenarios, and top-performing approaches are often a mix of techniques, which yields engineering-heavy solutions that lack generality. In this work, we question the need for hybrid approaches and introduce SUSHI, a unified and scalable multi-object tracker. Our approach processes long clips by splitting them into a hierarchy of subclips, which enables high scalability. We leverage graph neural networks to process all levels of the hierarchy, which makes our model unified across temporal scales and highly general. As a result, we obtain significant improvements over state-of-the-art on four diverse datasets.

<p align="center"><img src="assets/teaser.png" width="1200"/></p>




## Results
### MOTChallenge - Test Set
| Dataset    |  IDF1 | HOTA | MOTA | ID Sw. |
|------------|-------|------|------|-------|
|MOT17 - Public       | 71.5 | 54.6 | 62.0 | 1041 |
|MOT17 - Private      | 83.1 | 66.5 | 81.1 | 1149 |
|MOT20 - Public       | 71.6 | 55.4 | 61.6 | 1053 |
|MOT20 - Private      | 79.8 | 64.3 | 74.3 | 706 |

### DanceTrack - Test Set
| Dataset    |  IDF1 | HOTA | MOTA | AssA | DetA |
|------------|-------|------|------|-------|------|
|DanceTrack      | 63.4 | 63.3 | 88.7 | 50.1 | 80.1 |


### BDD - Test Set
| Dataset    |  mIDF1 | mMOTA | IDF1 | MOTA | ID Sw. |
|------------|-------|------|------|-------|------|
|BDD      | 60.0 | 40.2| 76.2 | 69.2 | 13626 |


## Setup



## Training 


## Testing




## Citation
If you use our work in your research, please cite our publication:

    @InProceedings{cetintas2023sushi,
        author={Orcun Cetintas and Guillem Brasó and Laura Leal-Taixé},
        title={Unifying Short and Long-Term Tracking with Graph Hierarchies},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2023}
    }



## Acknowledgements
We use the codebase of [TrackEval](https://github.com/JonathonLuiten/TrackEval) for evaluation. We thank the authors for their great work!