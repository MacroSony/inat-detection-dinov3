from custom_backbone import apply_monkey_patch

# 1. Inject the DINOv3 backbone into RF-DETR's registry
apply_monkey_patch()

# 2. Import RF-DETR trainer (must be AFTER the patch)
from rfdetr.training.cli import main

# 3. Start training using a custom YAML config
if __name__ == "__main__":
    main()
