
from .Baiyr import Baiyr


def build_model(config):
    """
    根据 YAML 配置构建模型实例。
    目前支持：
    - MODEL.NAME = "Baiyr"
    - FRONTEND_MODE: simple / ir50_single / ir50_multi / mobile_single / mobile_multi
    """

    model_name = config["MODEL"]["NAME"]
    num_classes = config["MODEL"]["NUM_CLASSES"]

    frontend_mode = config["MODEL"].get("FRONTEND_MODE", "simple")

    # 新增：分别指定 A / B 分支的模式
    # A 分支：a_simple / a_mobile_single / a_mobile_multi
    # B 分支：b_simple / b_ir50_single / b_ir50_multi
    branch_a_mode = config["MODEL"].get("BRANCH_A_MODE", "a_simple")
    branch_b_mode = config["MODEL"].get("BRANCH_B_MODE", "b_ir50_multi")
    bridge_out_channels = config["MODEL"].get("BRIDGE_OUT_CHANNELS", 32)
    bridge_use_multi_scale_residual = config["MODEL"].get("BRIDGE_USE_MULTI_SCALE_RESIDUAL", False)
    bridge_use_scale_weighting = config["MODEL"].get("BRIDGE_USE_SCALE_WEIGHTING", False)
    backend_type = config["MODEL"].get("BACKEND_TYPE", "transformer")
    light_backend_hidden_channels = config["MODEL"].get("LIGHT_BACKEND_HIDDEN_CHANNELS", 64)
    mamba_embed_dim = config["MODEL"].get("MAMBA_EMBED_DIM", 128)
    mamba_token_len = config["MODEL"].get("MAMBA_TOKEN_LEN", 49)
    mamba_num_layers = config["MODEL"].get("MAMBA_NUM_LAYERS", 2)
    mamba_d_state = config["MODEL"].get("MAMBA_D_STATE", 16)
    mamba_d_conv = config["MODEL"].get("MAMBA_D_CONV", 4)
    mamba_expand = config["MODEL"].get("MAMBA_EXPAND", 2)
    mamba_use_cls = config["MODEL"].get("MAMBA_USE_CLS", False)
    mamba_version = config["MODEL"].get("MAMBA_VERSION", "v1_plain")
    mamba_local_kernel_size = config["MODEL"].get("MAMBA_LOCAL_KERNEL_SIZE", 3)
    use_aux_head = config["MODEL"].get("USE_AUX_HEAD", False)

    ir50_pretrained_path = config["MODEL"].get("IR50_PRETRAINED_PATH", "")
    ir50_freeze = config["MODEL"].get("IR50_FREEZE", False)
    ir50_train_body2 = config["MODEL"].get("IR50_TRAIN_BODY2", False)
    ir50_train_last_stage = config["MODEL"].get("IR50_TRAIN_LAST_STAGE", False)

    mobile_pretrained_path = config["MODEL"].get("MOBILE_PRETRAINED_PATH", "")
    mobile_freeze = config["MODEL"].get("MOBILE_FREEZE", False)

    if model_name == "Baiyr":
        model = Baiyr(
            num_classes=num_classes,
            frontend_mode=frontend_mode,
            branch_a_mode=branch_a_mode,
            branch_b_mode=branch_b_mode,
            bridge_out_channels=bridge_out_channels,
            bridge_use_multi_scale_residual=bridge_use_multi_scale_residual,
            bridge_use_scale_weighting=bridge_use_scale_weighting,
            backend_type=backend_type,
            light_backend_hidden_channels=light_backend_hidden_channels,
            mamba_embed_dim=mamba_embed_dim,
            mamba_token_len=mamba_token_len,
            mamba_num_layers=mamba_num_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_use_cls=mamba_use_cls,
            mamba_version=mamba_version,
            mamba_local_kernel_size=mamba_local_kernel_size,
            use_aux_head=use_aux_head,
            ir50_pretrained_path=ir50_pretrained_path,
            ir50_freeze=ir50_freeze,
            ir50_train_body2=ir50_train_body2,
            ir50_train_last_stage=ir50_train_last_stage,
            mobile_pretrained_path=mobile_pretrained_path,
            mobile_freeze=mobile_freeze,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model