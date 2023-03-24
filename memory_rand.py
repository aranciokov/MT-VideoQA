import torch
import torch.nn as nn
import torch.nn.functional as F

     
class MemoryRamModule(nn.Module):

    def __init__(self, input_size=1024, hidden_size=512, memory_bank_size=100):
        """Set the hyper-parameters and build the layers."""
        super(MemoryRamModule, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_bank_size = memory_bank_size        
        
        self.hidden_to_content = nn.Linear(hidden_size+input_size, hidden_size)  
        #self.read_to_hidden = nn.Linear(hidden_size+input_size, 1)  
        self.write_gate = nn.Linear(hidden_size+input_size, 1)  
        self.write_prob = nn.Linear(hidden_size+input_size, memory_bank_size)  

        self.read_gate = nn.Linear(hidden_size+input_size, 1)  
        self.read_prob = nn.Linear(hidden_size+input_size, memory_bank_size)  

        # TODO ma questi FloatTEnsor richiedono veramente parametri?
        # TODO Ho letto che si puo' fare torch.cuda.FloatTensor per usare la GPU altrimenti si sta usando la CPU
        self.Wxh = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Wrh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Whh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bh = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)

        self.init_weights()
        
    def init_weights(self):
        self.Wxh.data.normal_(0.0, 0.01)
        self.Wrh.data.normal_(0.0, 0.01)
        self.Whh.data.normal_(0.0, 0.01)
        self.bh.data.fill_(0)

        
    def forward(self, hidden_frames, nImg, device):
                
        memory_ram = torch.FloatTensor(self.memory_bank_size, self.hidden_size).to(device)
        memory_ram.fill_(0)
        
        h_t = torch.zeros(1, self.hidden_size).to(device)
        
        hiddens = torch.FloatTensor(nImg, self.hidden_size).to(device)
                
        for t in range(nImg):
            x_t = hidden_frames[t:t+1,:]
            x_h_t = torch.cat([x_t,h_t],dim=1)
                        
            ############# read ############
            ar = torch.softmax(self.read_prob( x_h_t ),dim=1)  # read prob from memories
            go = torch.sigmoid(self.read_gate( x_h_t ))  # read gate
            r = go * torch.matmul(ar,memory_ram)  # read vector

            ######### h_t #########
            # Eq (17)
            m1 = torch.matmul(x_t, self.Wxh)
            m2 = torch.matmul(r, self.Wrh)
            m3 = torch.matmul(h_t, self.Whh)
            h_t_p1 = F.relu(m1 + m2 + m3 + self.bh)  # Eq(17)
            
            ############# write ############            
            c_t = F.relu( self.hidden_to_content(x_h_t) )  # Eq(15), content vector
            aw = torch.softmax(self.write_prob( x_h_t ),dim=1)  # write prob to memories
            aw = aw.view(self.memory_bank_size,1)
            gw = torch.sigmoid(self.write_gate( x_h_t ))  # write gate
            #print gw.size(),aw.size(),c_t.size(),memory_ram.size()
            memory_ram = gw * aw * c_t + (1.0-aw) * memory_ram # Eq(16)
            
            
            h_t = h_t_p1
            
            hiddens[t,:] = h_t

        

        #return memory_ram
        return hiddens

class MemoryRamModule_batch(nn.Module):

    def __init__(self, input_size=1024, hidden_size=512, memory_bank_size=100):
        """Set the hyper-parameters and build the layers."""
        super(MemoryRamModule_batch, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_bank_size = memory_bank_size

        self.hidden_to_content = nn.Linear(hidden_size+input_size, hidden_size)
        #self.read_to_hidden = nn.Linear(hidden_size+input_size, 1)
        self.write_gate = nn.Linear(hidden_size+input_size, 1)
        self.write_prob = nn.Linear(hidden_size+input_size, memory_bank_size)

        self.read_gate = nn.Linear(hidden_size+input_size, 1)
        self.read_prob = nn.Linear(hidden_size+input_size, memory_bank_size)

        self.Wxh = nn.Parameter(torch.FloatTensor(input_size, hidden_size),requires_grad=True)
        self.Wrh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.Whh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size),requires_grad=True)
        self.bh = nn.Parameter(torch.FloatTensor(hidden_size),requires_grad=True)

        self.init_weights()

    def init_weights(self):
        self.Wxh.data.normal_(0.0, 0.01)
        self.Wrh.data.normal_(0.0, 0.01)
        self.Whh.data.normal_(0.0, 0.01)
        self.bh.data.fill_(0)

    def forward(self, hidden_frames, nImg, device):
        batch_size = hidden_frames.shape[0]

        memory_ram = torch.FloatTensor(batch_size, self.memory_bank_size, self.hidden_size).to(device)
        memory_ram.fill_(0)

        h_t = torch.zeros(batch_size, 1, self.hidden_size).to(device)

        hiddens = torch.FloatTensor(batch_size, nImg, self.hidden_size).to(device)

        for t in range(nImg):
            x_t = hidden_frames[:, t:t+1, :]
            x_h_t = torch.cat([x_t, h_t], dim=-1)

            ############# read ############
            ar = torch.softmax(self.read_prob(x_h_t), dim=-1)  # read prob from memories
            #go = torch.sigmoid(self.read_gate(x_h_t))  # read gate
            r = torch.matmul(ar, memory_ram)  # read vector

            ######### h_t #########
            # Eq (17)
            m1 = torch.matmul(x_t, self.Wxh)
            m2 = torch.matmul(r, self.Wrh)
            m3 = torch.matmul(h_t, self.Whh)
            h_t_p1 = F.relu(m1 + m2 + m3 + self.bh)  # Eq(17)

            ############# write ############
            c_t = F.relu(self.hidden_to_content(x_h_t))  # Eq(15), content vector
            aw = torch.softmax(self.write_prob(x_h_t), dim=-1)  # write prob to memories
            aw = aw.view(batch_size, self.memory_bank_size, 1)
            #gw = torch.sigmoid(self.write_gate(x_h_t))  # write gate
            #print gw.size(),aw.size(),c_t.size(),memory_ram.size()
            memory_ram = aw * c_t + (1.0 - aw) * memory_ram  # Eq(16)

            h_t = h_t_p1

            hiddens[:, t:t+1, :] = h_t

        return hiddens


class MemoryRamTwoStreamModule(nn.Module):

    def __init__(self, input_size, hidden_size=512, memory_bank_size=100):
        """Set the hyper-parameters and build the layers."""
        super(MemoryRamTwoStreamModule, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_bank_size = memory_bank_size        
        
        self.hidden_to_content_a = nn.Linear(hidden_size+input_size, hidden_size)  
        self.hidden_to_content_m = nn.Linear(hidden_size+input_size, hidden_size)  

        self.write_prob = nn.Linear(hidden_size*3, 3)  
        self.write_prob_a = nn.Linear(hidden_size+input_size, memory_bank_size)  
        self.write_prob_m = nn.Linear(hidden_size+input_size, memory_bank_size)  

        self.read_prob = nn.Linear(hidden_size*3, memory_bank_size)  

        self.read_to_hidden = nn.Linear(hidden_size*2, hidden_size)  
        self.read_to_hidden_a = nn.Linear(hidden_size*2+input_size, hidden_size)  
        self.read_to_hidden_m = nn.Linear(hidden_size*2+input_size, hidden_size)

    def init_weights(self):
        pass
        
        
    def forward(self, hidden_out_a, hidden_out_m, nImg, device):
        
        
        memory_ram = torch.FloatTensor(self.memory_bank_size, self.hidden_size).to(device)
        memory_ram.fill_(0)
        
        h_t_a = torch.zeros(1, self.hidden_size).to(device)
        h_t_m = torch.zeros(1, self.hidden_size).to(device)
        h_t = torch.zeros(1, self.hidden_size).to(device)

        hiddens = torch.FloatTensor(nImg, self.hidden_size).to(device)
        
        for t in range(nImg):
            x_t_a = hidden_out_a[t:t+1,:]
            x_t_m = hidden_out_m[t:t+1,:]
            
                        
            ############# read ############
            x_h_t_am = torch.cat([h_t_a,h_t_m,h_t],dim=1)
            ar = torch.softmax(self.read_prob( x_h_t_am ),dim=1)  # read prob from memories
            r = torch.matmul(ar,memory_ram)  # read vector


            ######### h_t #########
            # Eq (17)
            f_0 = torch.cat([r, h_t],dim=1)
            f_a = torch.cat([x_t_a, r, h_t_a],dim=1)
            f_m = torch.cat([x_t_m, r, h_t_m],dim=1)
            
            h_t_1 = F.relu(self.read_to_hidden(f_0))
            h_t_a1 = F.relu(self.read_to_hidden_a(f_a))
            h_t_m1 = F.relu(self.read_to_hidden_m(f_m))
            
            
            ############# write ############            
            
            # write probability of [keep, write appearance, write motion]
            aw = torch.softmax(self.write_prob( x_h_t_am ),dim=1)  # write prob to memories
            x_h_ta = torch.cat([h_t_a,x_t_a],dim=1)
            x_h_tm = torch.cat([h_t_m,x_t_m],dim=1)
            
            
            # write content
            c_t_a = F.relu( self.hidden_to_content_a(x_h_ta) )  # Eq(15), content vector
            c_t_m = F.relu( self.hidden_to_content_m(x_h_tm) )  # Eq(15), content vector

            aw_a = torch.softmax(self.write_prob_a( x_h_ta ),dim=1)  # write prob to memories
            aw_m = torch.softmax(self.write_prob_m( x_h_tm ),dim=1)  # write prob to memories


            aw_a = aw_a.view(self.memory_bank_size,1)
            aw_m = aw_m.view(self.memory_bank_size,1)
            
            memory_ram = aw[0,0] * memory_ram + aw[0,1] * aw_a * c_t_a + aw[0,2] * aw_m * c_t_m
            
            
            h_t = h_t_1
            h_t_a = h_t_a1
            h_t_m = h_t_m1
            
            hiddens[t,:] = h_t

        
        return hiddens

class MemoryRamTwoStreamModule_batch(nn.Module):

    def __init__(self, input_size, hidden_size=512, memory_bank_size=100):
        """Set the hyper-parameters and build the layers."""
        super(MemoryRamTwoStreamModule_batch, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_bank_size = memory_bank_size

        self.hidden_to_content_a = nn.Linear(hidden_size+input_size, hidden_size)
        self.hidden_to_content_m = nn.Linear(hidden_size+input_size, hidden_size)

        self.write_prob = nn.Linear(hidden_size*3, 3)
        self.write_prob_a = nn.Linear(hidden_size+input_size, memory_bank_size)
        self.write_prob_m = nn.Linear(hidden_size+input_size, memory_bank_size)

        self.read_prob = nn.Linear(hidden_size*3, memory_bank_size)

        self.read_to_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.read_to_hidden_a = nn.Linear(hidden_size*2+input_size, hidden_size)
        self.read_to_hidden_m = nn.Linear(hidden_size*2+input_size, hidden_size)

    def init_weights(self):
        pass

    def forward(self, hidden_out_a, hidden_out_m, nImg, device):
        batch_size = hidden_out_a.shape[0]

        memory_ram = torch.FloatTensor(batch_size, self.memory_bank_size, self.hidden_size).to(device)
        memory_ram.fill_(0)

        h_t_a = torch.zeros(batch_size, 1, self.hidden_size).to(device)
        h_t_m = torch.zeros(batch_size, 1, self.hidden_size).to(device)
        h_t = torch.zeros(batch_size, 1, self.hidden_size).to(device)

        hiddens = torch.FloatTensor(batch_size, nImg, self.hidden_size).to(device)

        for t in range(nImg):
            x_t_a = hidden_out_a[:, t:t+1, :]
            x_t_m = hidden_out_m[:, t:t+1, :]

            ############# read ############
            x_h_t_am = torch.cat([h_t_a, h_t_m, h_t], dim=-1)  # (B, seq_len, 3*H)
            ar = torch.softmax(self.read_prob(x_h_t_am), dim=-1)  # read prob from memories
            r = torch.matmul(ar, memory_ram)  # read vector

            ######### h_t #########
            # Eq (17)
            f_0 = torch.cat([r, h_t], dim=-1)
            f_a = torch.cat([x_t_a, r, h_t_a], dim=-1)
            f_m = torch.cat([x_t_m, r, h_t_m], dim=-1)

            h_t_1 = F.relu(self.read_to_hidden(f_0))
            h_t_a1 = F.relu(self.read_to_hidden_a(f_a))
            h_t_m1 = F.relu(self.read_to_hidden_m(f_m))

            ############# write ############

            # write probability of [keep, write appearance, write motion]
            aw = torch.softmax(self.write_prob(x_h_t_am), dim=-1)  # write prob to memories
            x_h_ta = torch.cat([h_t_a, x_t_a], dim=-1)
            x_h_tm = torch.cat([h_t_m, x_t_m], dim=-1)

            # write content
            c_t_a = F.relu(self.hidden_to_content_a(x_h_ta))  # Eq(15), content vector
            c_t_m = F.relu(self.hidden_to_content_m(x_h_tm))  # Eq(15), content vector

            aw_a = torch.softmax(self.write_prob_a(x_h_ta), dim=-1)  # write prob to memories
            aw_m = torch.softmax(self.write_prob_m(x_h_tm), dim=-1)  # write prob to memories

            aw_a = aw_a.view(batch_size, self.memory_bank_size, 1)
            aw_m = aw_m.view(batch_size, self.memory_bank_size, 1)

            memory_ram = aw[:, 0, 0].unsqueeze(1).unsqueeze(1) * memory_ram + \
                         aw[:, 0, 1].unsqueeze(1).unsqueeze(1) * aw_a * c_t_a +\
                         aw[:, 0, 2].unsqueeze(1).unsqueeze(1) * aw_m * c_t_m

            h_t = h_t_1
            h_t_a = h_t_a1
            h_t_m = h_t_m1

            hiddens[:, t:t+1, :] = h_t

        return hiddens
