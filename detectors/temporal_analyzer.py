import torch
import torch.nn as nn

class TemporalAnalyzer(nn.Module):
    """
    CNN-LSTM network that analyzes sequence of face crops to detect 
    temporal inconsistencies (flickering, unnatural movement).
    """
    def __init__(self, hidden_size=256, num_layers=1):
        super(TemporalAnalyzer, self).__init__()
        import torchvision.models as models
        
        # Feature extractor
        resnet = models.resnet18(weights=None)
        # Remove FC layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1) # Output probability of fake

    def forward(self, x):
        # x shape: [batch, sequence_length, C, H, W]
        b, seq, c, h, w = x.size()
        x = x.view(b * seq, c, h, w)
        
        features = self.feature_extractor(x)
        features = features.view(b, seq, -1)
        
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]
        
        logits = self.fc(last_hidden)
        return torch.sigmoid(logits)

class TemporalFeatureExtractor:
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = TemporalAnalyzer().to(device)
        if model_path:
            # load weights
            pass
        self.model.eval()
        
    def extract_temporal_score(self, frame_tensors):
        """
        frame_tensors: list of [3, 512, 512] tensors
        Returns a float score.
        """
        if not frame_tensors:
            return 0.5
            
        with torch.no_grad():
            seq = torch.stack(frame_tensors).unsqueeze(0).to(self.device)
            # CNN-LSTM expects smaller images usually (e.g. 224x224), 
            # so we'd interpolate here
            seq = torch.nn.functional.interpolate(
                seq.view(-1, 3, 512, 512), size=(224, 224)
            ).view(1, len(frame_tensors), 3, 224, 224)
            
            score = self.model(seq).item()
        return score
