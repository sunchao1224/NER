# NER-BiLSTM-CRF-PyTorch
PyTorch implementation of BiLSTM-CRF models for named entity recognition.

## Requirements
- Python 3
- PyTorch 1.x

## Dataset
- CoNLL 2003 (English)
- E-NER Dataset (Legal domain)

### Evaluation
- conlleval: Perl script used to calculate FB1 (**phrase level**)

## Model
- Embeddings
  - 100d pre-trained word embedding with Glove
  - 25d charactor embedding trained by CNNs (Ma et al., 2016)
- BiLSTM-CRF (Lample et. al., 2016)

## Visdom
1. Install Visdom on the remote server (if it isn't already installed):

```bash
pip install visdom
```

2. Start the Visdom server on the remote machine. To make it accessible from external IPs, use the `--hostname` flag with `0.0.0.0`:

```bash
visdom -port 8097 --hostname 0.0.0.0
```

Now, the Visdom server will be accessible on port 8097 from any IP address.

3. To access the Visdom server from your local machine, open a web browser and go to the following URL (replace `your_remote_server_ip` with the actual IP address of your remote server):

```
http://your_remote_server_ip:8097
```

4. In your Python code running on the remote server, make sure to set the Visdom server's IP address and port when creating the Visdom instance:

```python
import visdom

vis = visdom.Visdom(server='http://your_remote_server_ip', port=8097)
```

Replace `your_remote_server_ip` with the actual IP address of your remote server.

5. Now, your Python code running on the remote server should be able to send data to the Visdom server, and you can view the visualizations on your local machine by visiting the URL mentioned in step 3.

## Usage:
`conda activate bilstm+crf`

`python train.py` to train the model.

`python transfer.py` to transfer parameters.

`python self-training.py` to do self-training.

`streamlit run streamlit_app.py` to open the demo(connect to gpu server first).

![Alt Text](images/demo.gif)



## References
- https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
- https://github.com/ZubinGou/NER-BiLSTM-CRF-PyTorch
