# [WWW 2025] JobFed: Joint Optimization for Balancing Global and Local Personalized Models in Hierarchical Federated Learning

Supplementary Materials for WWW 2025 [JobFed: Joint Optimization for Balancing Global and Local Personalized Models in Hierarchical Federated Learning] by *Xiangchi Song, Arogya Kharel, Eunkyoung Jee*, and *In-Young Ko*. School of Computing,  Korea Advanced Institute of Science and Technology,  Daejeon, Republic of Korea.

## Table of Contents
* Overview
* System Code & Experimental Records
* Supplementary Material 1: Mathematical representation of the complete model
* Supplementary Material 2: Sequence diagram of the architecture workflow
* Model & Parameter Explanation
* Dataset Distribution Operation Detail
* Contact
* Special Thanks
* References

## Overview

In today's large-scale Internet of Things (IoT) environments, Federated Learning (FL) has demonstrated exceptional performance in simultaneously addressing the demands for data utilization and user privacy protection. To reduce the substantial communication overhead of current FL systems, hierarchical FL architecture has emerged as a promising solution. 
However, due to the technical differences in training global models and fine-tuning personalized models, most existing hierarchical FL methods primarily focus on either constructing a better-performing global model or customizing personalized models for regions or users, with limited research addressing the balance between global and local personalized models. We have designed a cloud-fog-edge hierarchical interaction architecture based on FL, establishing both global and local models while significantly improving the system's interaction performance.By utilizing model information exchanged between layers in the architecture, we formulate the balance between global and local personalized models as a joint optimization problem to achieve relatively optimal performance for both. Experimental results demonstrate the effectiveness of our hierarchical interaction-based joint optimization approach.

### System Code & Experimental Records
***About system Code & experimental records, we will make it public after the paper is published.***

#### System Requirements
- **python**: `3.9`
- **torch**: `2.1.1`  **cuda**: `12.1`  **cudnn**: `8.0`  **torchvision**: `0.8.0`  
- **numpy**: `1.24.1`  **pandas**: `2.2.3`  **progressbar2**: `2.5`  **tqdm**: `4.46.2`

## Supplementary Material 1: Mathematical representation of the complete model
***Please refer to the [Appendix.pdf](https://github.com/XiangchiSong/WWW2025_JOB-Fed/blob/main/Appendix.pdf)***

## Supplementary Material 2: Sequence diagram of the architecture workflow

<table>
  <tr>
    <td align="center" valign="middle" width="60%">
      <img src="https://raw.githubusercontent.com/XiangchiSong/WWW2025_JOB-Fed/main/SequenceWorkflow.png" alt="Sequence Workflow" width="700">
    </td>
    <td valign="middle" width="50%">
      <div style="font-size:70%;">

**Process Overview**

The process begins with the *edge* devices initiating communication with the *fog* layer by sending identification information (`sendIDInfo`). In response, the *fog* layer provides the *edge* devices with initial cluster configurations (`sendClusterConfig`).

This initial clustering is done randomly, and the *edge* devices are assigned to clusters that are internally connected based on the P2P connection.

Concurrently, the *cloud* initializes the global model (`initGlobalModel`) and distributes the initial model to the *fog* layer, laying the foundation for subsequent local training of client models on the *edge* devices and aggregation within the *fog* and *cloud* layers.

The processes of initial clustering for *edge* devices and the model initialization in the *cloud* server are performed in parallel.

Upon receiving the initial global model from the *cloud* server, the *fog* layers distribute the global model to all of the *edge* devices they are responsible for. The objective here is to train the global model on local data available at the *edge* device. This training process occurs in a loop, iteratively refining the global model and the local models until both reach a certain level of convergence and performance.

Once the distribution from the *fog* layer to the *edge* devices is complete, the *edge* devices begin local training (`beginLocalTraining(initModel)`), updating the model based on local private data. After a round of training is completed, the private information is separated from the trained model (`separatePrivateInfo`) to create a personalized and non-personalized model. The private information is represented by separable parameters stored in Batch Normalization (BN) layers.

Both models are then concurrently sent to the *fog* layer (`sendPModel`, `sendNPModel`) for further aggregation (`beginFogAggregation`). The aggregation at the *fog* layers enhances the models' generalization across the data from various *edge* devices while maintaining privacy. Upon completing the aggregation, the *fog* layer will perform reclustering (`sendClusterConfig`) for the clients corresponding to the *edge* devices, reallocating clusters based on the personalized model update directions of different clients to group similar models together (for the rationale and technical details of reclustering, please refer to the *Supplementary Materials*).

Simultaneously, the aggregated models are transmitted to the *cloud* server (`sendPModel`, `sendNPModel`), where the *cloud* server performs the global aggregation (`beginAggregation`) for both models. This step integrates the insights from all *fog* layers, resulting in a refined global model that captures the collective knowledge of the entire system. Once the aggregation is complete, the *cloud* server stores the global model, updating and overwriting it after each training round. This process runs in parallel with the aforementioned computation on the *fog* layer and the reclustering of *edge* devices.

Subsequently, the cloud sends the globally aggregated non-personalized model back to the *fog* layer (`sendGlobalNPModel`), which then distributes it to all *edge* devices represented by clients. This model contains public information from all clients on their respective devices, making it more suitable for local personalized adjustments later. Upon receiving this model, the client combines it with its local private information (`combinePrivateInfo`) to execute a personalization process. During the personalization process, the BN layers adjust each client’s local model, allowing the global model to adapt to local features. The specific operations of the BN layers and their role in balancing between the global model and local personalized models are detailed in Section \ref{sec:JOmodel}.

Afterward, the client uses the newly personalized model to start the local training again (`beginLocalTraining(updatedPModel)`). Once the local training is completed, the new model is retained and uploaded to the *fog* layer. The private information is separated again, resulting in a new non-personalized model. This private information is stored locally on the *edge* devices and is updated and overwritten after being separated at the end of each training round.

This process is then repeated in a loop until the appropriate convergence or stopping condition is met.

  </tr>
</table>

#### Classification Structure
- A fully connected (FC) layer with 200 neurons.
- A batch normalization (BN) layer.
- A second FC layer also with 200 neurons.
- A softmax output layer to generate the final classification probabilities.
#### CNN Model 
Following the methodologies employed in the MTFL[2] work, the CNN architecture includes:
- A 3x3 convolutional (conv) layer with 32 filters, followed by BN, ReLU activation, and a 2x2 max pooling.
- A second 3x3 convolutional ReLU layer with 64 filters, accompanied by BN, ReLU activation, and a 2x2 max pooling.
- A ReLU activated FC layer with 512 neurons.
- A softmax output layer for classification.
#### Model communication & computation Parameter Settings
EPFLU sets the storage space occupied by the number of parameters of the deep learning model as the model communication size:
- `comm_datasize = 6400320 bit` representing the MINST Model size (approximately 781.29 KB) in bits.
- `comm_datasize = 67758080 bit` representing the CIFAR Model size (approximately 8273 KB or 8.1 MB) in bits.

EPFLU considers the amount of data involved in processing a single iteration or single data sample as the local computation size:
- `local_datasize = 6272 bit` for MINST data size (784 B) in bits.
- `local_datasize = 24576 bit` for CIFAR data size (approximately 3072 B or 3 KB) in bits.

### EPFLU-P2PFL Parameter Settings
The configuration for EPFLU-P2PFL is specified as follows:
- **Data Distribution Type (iid)**: Options include `1` for balanced-iid, `2` for balanced-non-iid, `3` for imbalanced-non-iid, and `4` for imbalanced-mixed-iid.
- **Algorithm (alg)**: Choices are `fedavg`, `fedadam`, `ppt`, `epflu`.
- **Total Workers (W)**: `500`, suitable for a large-scale IoT scenario.
- **Total Rounds (T)**: `300` rounds for CIFAR10 to verify performance and convergence, `50` rounds for MNIST to validate communication.
- **Sampling Rate (C)**: `0.3`, the best sampling rate after balancing training and communication costs under the large-scale IoT environment setting.
- **Client Learning Rate (lr)**: Different settings for different algorithms and datasets, e.g., `0.02` for CIFAR10 and `0.2` for MNIST with FedAvg; `0.05` for CIFAR10 and `0.2` for MNIST with FedAdam; `0.03` for CIFAR10 and `0.2` for MNIST with PPT; `0.001` for CIFAR10 and `0.001` for MNIST with EPFLU;
  
### Communication Simulator Parameter Settings
The simulation setup for communication parameters is as follows:
- **Bandwidth (B)**: `20 MHz` (2e7 Hz), representing typical bandwidths for LTE, 5G, and Wi-Fi.
- **Reference Channel Gain (g0)**: `1e-8`, representing the gain at a distance of one unit.
- **Transmitter Power (p)**: `0.5 Watts`.
- **Noise Power (sigma)**: `1e-10 Watts`.
- **Number of CPU Cycles per Bit (c)**: `10800`. This metric is adapted from GPU's "floating-point operations per second" (FLOPS) to simulate the scenario using CPUs for a large-scale IoT environment.
- **CPU Frequency (f)**: `1 GHz` (1e9 cycles per second), configured for an i7-9700K CPU at 3.60 GHz.
- **Effective Switch Capacitance (alpha)**: `2e-28` Joules per cycle squared.
- **Distance Threshold**: `200`, ensuring the reference channel gain does not drop to zero.
- **Communication to Cloud Cost Multiplier**: `1.5`, assuming that vertical FL communication costs are 1.5 times that of horizontal P2P communication. *In previous For the communication latency to the cloud, **HierFAVG**[5] assume it is **10** times larger than that to the edge.*
- **Dynamic Multiplier**: Adjusted based on a reference average distance of `50` units.
- **Extra Loss**: `10`, this is an assumed value, adjusted based on actual conditions.
- **Minimum Channel Gain to Cloud (min_g_2C)**: `5e-11`.
- **Minimum Channel Gain P2P (min_g_P2P)**: `1e-10`.
- **Distance Matrix**: A random distance matrix representing the distances between edge clients, ranging from `1 to 100`.

## Dataset Distribution Operation Detail
### Core Concepts
- **Data Shards**: A shard represents a subset of the dataset. Managing the number and distribution of shards across clients is crucial for controlling the IID (independent and identically distributed) or non-IID nature of data across the network. 

### Parameter Settings and Their Impact
- **Number of Shards (n_shards)**: Dictates how the data is divided. More shards mean a finer granularity in control over data distribution.
- **Number of Workers (W)**: Represents the total number of clients or nodes participating in the federated learning. Each worker will receive a portion of the total shards.
- **Shard Allocation**:
  - **Balanced**: Each client receives an equal number of shards.
  - **Imbalanced**: Clients receive a varying number of shards, which simulates real-world scenarios where data isn't uniformly distributed across nodes.

### Operational Details
- **Shard Assignment**:
  - For IID setups, shards are assigned randomly.
  - For non-IID setups, shards are assigned based on specific patterns or rules, such as clustering similar classes together or ensuring that each client receives shards from only a few classes.
- **Min and Max Shards per Client**: These parameters define the range of shards that each client can receive, crucial for imbalanced setups.
  
### Distribution Types
- **Balanced IID (Type 1)**: Data is shuffled and evenly distributed among all clients, mimicking an IID scenario. This is often used for benchmarking because of its simplicity and fairness.
- **Balanced Non-IID (Type 2)**: Involves sorting the data by class labels and then splitting it into shards that are distributed such that each client receives a predefined set of shards. This setup helps in creating scenarios where each client gets a diverse but non-representative sample of the overall dataset. We firstly sorted by class labels and then divided into `2 * W` shards.
- **Imbalanced Non-IID (Type 3)**: Clients receive shards in a manner that varies significantly in terms of data quantity and class representation, thus creating an imbalanced data distribution among the clients. This setting is more realistic and challenging as it tests the robustness of the federated algorithms under non-ideal conditions. In this configuration, the number of shards each client receives can range from 1 to 5, creating a highly variable data distribution. We used the parameters `min_shards_per_client = 1` and `max_shards_per_client = 2` in the EPFLU settings.
- **Imbalanced Mixed-IID (Type 4)**: A hybrid approach where a fraction of the clients receive data in an IID fashion, while the rest receive data non-IID. This setup can be used to simulate environments where different nodes have different data visibility and availability. We used `50%` of clients (calculated as `0.5 * W`) receive data in an IID distribution, and the remaining `50%` receive data non-IID. The non-IID portion follows the imbalanced non-IID strategy. After we created the Mixed-IID scenario, We used the parameters `min_shards_per_client = 1` and `max_shards_per_client = 2` to create an imbalanced amount of data between clients, just like Type3 does in EPFLU.


### Implementation Tips
- Ensuring consistency in shard assignment between training and testing datasets `specific_assignments=train_assign` is critical for maintaining the validity of the model evaluation.
- Using seed values for random operations helps in reproducing experiments and verifying results. We used `42` in imbalanced_non_iid and imbalanced_mixed_iid scenarios.

The configuration of these parameters significantly influences the learning dynamics and the effectiveness of the federated learning models. Adjusting them according to the specific needs of the deployment scenario can lead to better model performance and more robust insights.


## Experimental Records

## Contact
If you like our works, please cite our paper. Also, feel free to contact us: xcsong@kaist.ac.kr, we will reply to you within three working days！

## Special Thanks

## References
[1] McMahan B, Moore E, Ramage D, et al. [Communication-efficient learning of deep networks from decentralized data](https://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com)[C]//Artificial intelligence and statistics. PMLR, 2017: 1273-1282.

[2] Mills J, Hu J, Min G. [Multi-task federated learning for personalised deep neural networks in edge computing](https://ieeexplore.ieee.org/abstract/document/9492755)[J]. IEEE Transactions on Parallel and Distributed Systems, 2021, 33(3): 630-641.

[3] Reddi S, Charles Z, Zaheer M, et al. [Adaptive federated optimization](https://arxiv.org/abs/2003.00295)[J]. arXiv preprint arXiv:2003.00295, 2020.

[4] Chen Q, Wang Z, Zhang W, et al. [PPT: A privacy-preserving global model training protocol for federated learning in P2P networks](https://www.sciencedirect.com/science/article/pii/S0167404822003583)[J]. Computers & Security, 2023, 124: 102966.

[5] Liu L, Zhang J, Song S H, et al. [Client-edge-cloud hierarchical federated learning](https://ieeexplore.ieee.org/abstract/document/9148862)[C]//ICC 2020-2020 IEEE international conference on communications (ICC). IEEE, 2020: 1-6.

## 
Copyright © 2024 Xiangchi Song, Zhaoyan Wang, KyeongDeok Baek, and In-Young Ko

This research was partly supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2024-2020-0-01795) supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation) and IITP grant funded by the Korea government (MSIT) (No. RS-2024-00406245, Development of Software-Defined Infrastructure Technologies for Future Mobility).

All rights reserved. No part of this publication may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the publisher, except in the case of brief quotations embodied in critical reviews and certain other noncommercial uses permitted by copyright law. For permission requests, please email to the author.
