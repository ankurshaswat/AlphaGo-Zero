from NNet import NNet

if __name__ == "__main__":
    net = NNet()
    net.save_checkpoint(path='model.pytorch')
