import numpy as np
import pandas as pd
import torch
import time
from torch.nn import functional
from tracker import ModelPerformanceTracker
from tqdm import tqdm

def train_and_validate(model, train_loader, valid_loader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    total_loss = 0

    # Training loop
    for sequences, targets in train_loader:
        # Move sequences and targets to device (GPU if available)
        sequences, targets = sequences.to(device), targets.to(device)
        # Forward pass and loss calculation
        optimizer.zero_grad()
        outputs = model(sequences)

        loss = functional.mse_loss(outputs, targets)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_train_loss = total_loss / len(train_loader)

    # Validation step
    model.eval()  # Set the model to evaluation mode
    total_valid_loss = 0

    with torch.no_grad():  # No gradient calculation during validation
        for sequences, targets in valid_loader:
            sequences, targets = sequences.to(device), targets.to(device)

            outputs = model(sequences)
            loss = functional.mse_loss(outputs, targets)
            total_valid_loss += loss.item()

    average_valid_loss = total_valid_loss / len(valid_loader)

    return average_train_loss, average_valid_loss


# TODO scaler 처리 주식 점진적으로 증가할 때 어떻게 할건지

def test(model, test_loader, scaler_X, scaler_y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions, actuals, dates = [], [], []  # 날짜 리스트 추가

    with torch.no_grad():
        for sequences, targets, batch_dates in test_loader:  # 날짜도 함께 로드
            # Move sequences and targets to device
            sequences, targets = sequences.to(device), targets.to(device)

            outputs = model(sequences)

            # 결과 저장
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

            # 날짜는 타임스탬프이므로 그대로 추가
            dates.extend(batch_dates.cpu().numpy())

    # Inverse transform for scaling
    predictions = scaler_y.inverse_transform(np.array(predictions))
    actuals = scaler_y.inverse_transform(np.array(actuals))

    # 날짜를 datetime으로 변환
    dates = pd.to_datetime(dates, unit='s')  # 타임스탬프에서 datetime으로 변환

    return predictions, actuals, dates

def train_and_test(model, optimizer, data, epochs, model_name, tracker: ModelPerformanceTracker):
    train_loader, valid_loader, test_loader, scaler_X, scaler_y = data
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Progress bar 생성
    with tqdm(total=epochs, desc=f"Training {model_name}", unit="epoch") as pbar:
        for epoch in range(epochs):
            start_time = time.time()
            train_loss, valid_loss = train_and_validate(model, train_loader, valid_loader, optimizer)
            epoch_time = time.time() - start_time

            tracker.update(model_name, epoch, train_loss, valid_loss, epoch_time, params_count)

            # Progress bar 업데이트
            pbar.set_postfix({
                'Train Loss': f"{train_loss:.4f}",
                'Valid Loss': f"{valid_loss:.4f}",
                'Epoch Time (s)': f"{epoch_time:.2f}"
            })
            pbar.update(1)

    # 테스트 단계
    predictions, actuals, result_dates = test(model, test_loader, scaler_X, scaler_y)
    tracker.save_test_results(model_name, predictions, actuals, result_dates)

    return predictions, actuals, result_dates

