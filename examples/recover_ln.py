from eml_lab.training import TrainConfig, train_target

if __name__ == "__main__":
    result = train_target(TrainConfig(target="ln", depth=3, seed=0, steps=180))
    print(result.rpn)
    print(result.verification.to_dict())
