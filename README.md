# [IJCAI 2025] JobFed: Joint Optimization for Balancing Generalization and Personalization of Hierarchical Federated Learning in Large-scale IoT Environments

## Table of Contents
* Supplementary Material 1 - Appendix: Technical details and results analysis of the paper.
* Supplementary Material 2 - Data Records: Data records in .pkl format for all experiments in this paper.


<!--Appendix and Data Records in Supplementary Materials for IJCAI 2025 [JobFed: Joint Optimization for Balancing Generalization and Personalization of Hierarchical Federated Learning in Large-scale IoT Environments] by *Xiangchi Song, Arogya Kharel, Eunkyoung Jee*, and *In-Young Ko*. School of Computing,  Korea Advanced Institute of Science and Technology,  Daejeon, Republic of Korea.

## Table of Contents
* Overview
* System Code & Experimental Records
* Supplementary Material 1: Mathematical representation of the complete model
* Supplementary Material 2: Sequence diagram of the architecture workflow
* Model & Parameter Explanation
* Dataset Distribution
* Contact
* Special Thanks
* References

## Overview
In today's large-scale Internet of Things (IoT) environments, Federated Learning (FL) has demonstrated exceptional performance in simultaneously addressing the demands for data utilization and user privacy protection. To reduce the substantial communication overhead of current FL systems, hierarchical FL architecture has emerged as a promising solution. 
However, due to the technical differences in training global models and fine-tuning personalized models, most existing hierarchical FL methods primarily focus on either constructing a better-performing global model or customizing personalized models for regions or users, with limited research addressing the balance between global and local personalized models. We have designed a cloud-fog-edge hierarchical interaction architecture based on FL, establishing both global and local models while significantly improving the system's interaction performance.By utilizing model information exchanged between layers in the architecture, we formulate the balance between global and local personalized models as a joint optimization problem to achieve relatively optimal performance for both. We choose FedAvg[1], pFedMe[2] as baselines. Experimental results demonstrate the effectiveness of our hierarchical interaction-based joint optimization approach.

## System Code & Experimental Records
***About system Code & experimental records, we will make it public after the paper is published.***

#### System Requirements
- **python**: `3.9`
- **torch**: `2.1.1`  **cuda**: `12.1`  **cudnn**: `8.0`  **torchvision**: `0.8.0`  
- **numpy**: `1.24.1`  **scipy**: `1.12.0`  **pandas**: `2.2.3`  **progressbar2**: `2.5`  **tqdm**: `4.46.2`

## Supplementary Material 1: Mathematical representation of the complete model
***Please refer to the [Appendix.pdf](https://github.com/XiangchiSong/WWW2025_JOB-Fed/blob/main/Appendix.pdf)***

## Supplementary Material 2: Sequence diagram of the architecture workflow
<table>
  <tr>
    <td align="center" valign="middle" width="40%">
      <img src="https://raw.githubusercontent.com/XiangchiSong/WWW2025_JOB-Fed/main/SequenceWorkflow.png" alt="Sequence Workflow" width="600">
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

Both models are then concurrently sent to the *fog* layer (`sendPModel`, `sendNPModel`) for further aggregation (`beginFogAggregation`). The aggregation at the *fog* layers enhances the models' generalization across the data from various *edge* devices while maintaining privacy. Upon completing the aggregation, the *fog* layer will perform reclustering (`sendClusterConfig`) for the clients corresponding to the *edge* devices, reallocating clusters based on the personalized model update directions of different clients to group similar models together.

Simultaneously, the aggregated models are transmitted to the *cloud* server (`sendPModel`, `sendNPModel`), where the *cloud* server performs the global aggregation (`beginAggregation`) for both models. This step integrates the insights from all *fog* layers, resulting in a refined global model that captures the collective knowledge of the entire system. Once the aggregation is complete, the *cloud* server stores the global model, updating and overwriting it after each training round. This process runs in parallel with the aforementioned computation on the *fog* layer and the reclustering of *edge* devices.

Subsequently, the cloud sends the globally aggregated non-personalized model back to the *fog* layer (`sendGlobalNPModel`), which then distributes it to all *edge* devices represented by clients. This model contains public information from all clients on their respective devices, making it more suitable for local personalized adjustments later. Upon receiving this model, the client combines it with its local private information (`combinePrivateInfo`) to execute a personalization process. During the personalization process, the BN layers adjust each client’s local model, allowing the global model to adapt to local features.

Afterward, the client uses the newly personalized model to start the local training again (`beginLocalTraining(updatedPModel)`). Once the local training is completed, the new model is retained and uploaded to the *fog* layer. The private information is separated again, resulting in a new non-personalized model. This private information is stored locally on the *edge* devices and is updated and overwritten after being separated at the end of each training round.

This process is then repeated in a loop until the appropriate convergence or stopping condition is met.

  </tr>
</table>

## Model & Parameter Explanation
JobFed uses most of the models and parameter settings of EPFLU[3], please [click here](https://github.com/XiangchiSong/EPFLU_P2PFL?tab=readme-ov-file#model--parameter-settings) for more details. In addition, we use the following parameters:
- **α<sub>i</sub>**: Mixing parameter; dynamically controls the balance between the global model and the local personalized models.
- **J<sub>global</sub>(g<sup>*</sup>)**: Loss function of the global model.
- **J<sub>local</sub>(l<sub>x</sub><sup>*</sup>)**: Loss function of the local personalized models for client *x*.
- **R<sub>i</sub>**: Relative optimal control condition; combines loss proportion and loss change rate.
- **β**: Weighting parameter in the Relative Optimal Control Condition; balances the importance between loss ratio and loss change rate in R<sub>i</sub>, the default setting is `0.5`.
- **ΔJ<sub>global</sub>**: Change rate of the global model's loss (e.g., current loss relative to the previous round).
- **ΔJ<sub>local</sub>**: Change rate of the personalized models' loss.
- **η**: Learning rate; controls the adjustment step for α, the default setting is `0.1`. 
- **γ**: Preset balance ratio; serves as the ideal value for R<sub>i</sub>, the default setting is `0.5`. 

## Dataset Distribution
JobFed uses similar dataset distribution operations with EPFLU, please [click here](https://github.com/XiangchiSong/EPFLU_P2PFL?tab=readme-ov-file#dataset-distribution-operation-detail) for more details.

## Contact
If you like our works, please cite our paper. Also, feel free to contact us: xcsong@kaist.ac.kr, we will reply to you within three working days！

## Special Thanks
We would like to thank [Jed Mills](https://scholar.google.com/citations?user=30_1nBcAAAAJ&hl=zh-CN&oi=sra) for providing the personalized scheme based on the BN Patch Layer mechanism[4], and [Yuyang Deng](https://sites.psu.edu/yuyangdeng/) for inspiring the Optimal Mixing Parameter method[5].

## References
[1] McMahan B, Moore E, Ramage D, et al. [Communication-efficient learning of deep networks from decentralized data](https://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com)[C]//Artificial intelligence and statistics. PMLR, 2017: 1273-1282.

[2] T Dinh C, Tran N, Nguyen J. [Personalized federated learning with moreau envelopes](https://proceedings.neurips.cc/paper/2020/hash/f4f1f13c8289ac1b1ee0ff176b56fc60-Abstract.html)[J]. Advances in neural information processing systems, 2020, 33: 21394-21405.

[3] Xiangchi Song, Zhaoyan Wang, KyeongDeok Baek, and In-Young Ko. Epflu: Efficient peer-to-peer federated learning for personalized user models in edge-cloud environments. The 4th International Workshop on Big Data Driven Edge Cloud Services (BECS 2024), Co-located with the 24th International Conference on Web Engineering (ICWE 2024), June 17-20, 2024, Tampere, Finland, June 2024.

[4] Mills J, Hu J, Min G. [Multi-task federated learning for personalised deep neural networks in edge computing](https://ieeexplore.ieee.org/abstract/document/9492755)[J]. IEEE Transactions on Parallel and Distributed Systems, 2021, 33(3): 630-641.

[5] Deng Y, Kamani M M, Mahdavi M. [Adaptive personalized federated learning](https://arxiv.org/abs/2003.13461)[J]. arXiv preprint arXiv:2003.13461, 2020.

## 
Copyright © 2024 Xiangchi Song, Arogya Kharel, Eunkyoung Jee, and In-Young Ko

This research was partly supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2024-2020-0-01795) supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation) and IITP grant funded by the Korea government (MSIT) (No. RS-2024-00406245, Development of Software-Defined Infrastructure Technologies for Future Mobility).

All rights reserved. No part of this publication may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the publisher, except in the case of brief quotations embodied in critical reviews and certain other noncommercial uses permitted by copyright law. For permission requests, please email to the author.-->
