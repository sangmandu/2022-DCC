import torch
import torch.nn as nn
batch_size = 50
learning_rate = 0.0002
num_epoch = 100
def conv_block_1(in_dim,out_dim, activation,stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model


def conv_block_3(in_dim,out_dim, activation, stride=1):

    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model

class BottleNeck(nn.Module):
    def __init__(self,in_dim,mid_dim,out_dim,activation,down=False):
        super(BottleNeck,self).__init__()
        self.down=down
        
        # 특성지도의 크기가 감소하는 경우
        if self.down:
            self.layer = nn.Sequential(
              conv_block_1(in_dim,mid_dim,activation,stride=2),
              conv_block_3(mid_dim,mid_dim,activation,stride=1),
              conv_block_1(mid_dim,out_dim,activation,stride=1),
            )
            
            # 특성지도 크기 + 채널을 맞춰주는 부분
            self.downsample = nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=2)
            
        # 특성지도의 크기가 그대로인 경우
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim,mid_dim,activation,stride=1),
                conv_block_3(mid_dim,mid_dim,activation,stride=1),
                conv_block_1(mid_dim,out_dim,activation,stride=1),
            )
            
        # 채널을 맞춰주는 부분
        self.dim_equalizer = nn.Conv2d(in_dim,out_dim,kernel_size=1)
                  
    def forward(self,x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out

# 50-layer
class ResNet(nn.Module):

    def __init__(self, base_dim=32, num_classes=10):
        super(ResNet, self).__init__()
        self.activation = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3,base_dim,7,2,3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim,base_dim,base_dim*4,self.activation),
            BottleNeck(base_dim*4,base_dim,base_dim*4,self.activation),
            BottleNeck(base_dim*4,base_dim,base_dim*4,self.activation,down=True),
        )   
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim*4,base_dim*2,base_dim*8,self.activation),
            BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.activation),
            BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.activation),
            BottleNeck(base_dim*8,base_dim*2,base_dim*8,self.activation,down=True),
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim*8,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),            
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation),
            BottleNeck(base_dim*16,base_dim*4,base_dim*16,self.activation,down=True),
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim*16,base_dim*8,base_dim*32,self.activation),
            BottleNeck(base_dim*32,base_dim*8,base_dim*32,self.activation),
            BottleNeck(base_dim*32,base_dim*8,base_dim*32,self.activation),
        )
        self.avgpool = nn.AvgPool2d(1,1) 
        self.fc_layer = nn.Linear(4096,num_classes)
        
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(-1,4096)
        out = self.fc_layer(out)
        
        return out