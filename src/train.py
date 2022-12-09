from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import Dataset

scan_list = ["abc1", "abc2", "abc3", "abc4"]
label_list = ["1", "2", "3", "4"]

train_dataset = Dataset(scan_list, label_list)

train_loader = DataLoader(dataset=train_dataset, batch_size=2)

for inputs, targets in tqdm(train_loader):
    print("inputs ", inputs, " targets", targets )



