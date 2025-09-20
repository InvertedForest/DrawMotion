_base_ = ['../_base_/datasets/kit_ml_bs128.py']


dataset_name = 'kit_ml'
point_len=64
feat_dim=256
norm_pose_dim=21*3
stick_set = dict(
    train=dict(
        batch_size=512,
        epochs=50,
        lr=1e-4,
        workers=80,
        dataset_name=dataset_name,
    ),
    stickman_encoder=dict(
        point_len=point_len,
        in_dim=point_len*2,
        out_dim=feat_dim,
        d_model=512,
        dropout=0.1,
        activation='relu',
        nhead=16,
        num_layers=5,
        ff_dim=1024,
    ),
    stickman_decoder=dict(
        in_dim=feat_dim,
        out_dim=norm_pose_dim,
        fcn_dims=[512, 512, 512, 512, 512],
        # bran_dims=[512, 512],
        dropout=0.1,
        candidate_num=4,
    ),
    motion_encoder=dict(
        in_dim=251,
        out_dim=feat_dim,
        fcn_dims=[512, 512, 512, 512, 512],
        dropout=0.1
    ),
    loss=dict(
        loss1=1.,
        loss2=1.,
        loss3=1.
    )
)

# fp16
# fp16 = dict(loss_scale=512.)
# checkpoint saving
checkpoint_config = dict(interval=5)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='Adam', lr=2e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='CosineAnnealing', min_lr_ratio=0, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=60)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

input_feats = 251
locus_dim = 10
max_seq_len = 196
latent_dim = 512
time_embed_dim = 2048
text_latent_dim = 256
ff_size = 1024
num_heads = 8
dropout = 0.
index_num = 3
stickman_encoder_path = 'stickman/weight/kit_ml/split_weight/stickman_encoder.ckpt'
stickman_decoder_path = 'stickman/weight/kit_ml/split_weight/stickman_decoder.ckpt'
stickman_all_path = 'stickman/logs/kit_ml/fix_init/last.ckpt'

# model settings
model = dict(
    type='MotionDiffusion',
    loss_weight=dict(
        # motion_w=0,
        # index_w=0
        stickman_w=1.0,
        locus_w=1.0,
    ),
    guidance=dict(
        repeat=50,
        layer_num=3,
        scale=50,
        locus_w=1.0,
        stick_w=0.0,
        # manual=False,
        manual=True,
    ),
    index_num=index_num,
    motion_crop=[4, 4+20*9],
    model=dict(
        type='ReMoDiffuseTransformer',
        input_feats=input_feats,
        max_seq_len=max_seq_len,
        latent_dim=latent_dim,
        time_embed_dim=time_embed_dim,
        num_layers=4,
        condition_cfg=dict(
            text_p=0.7,
            stick_p=0.7,
            index_train_p = 0.7
        ),
        index_num=index_num,
        ca_block_cfg=dict(
            type='SemanticsModulatedAttention',
            latent_dim=latent_dim,
            text_latent_dim=text_latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            locus_dim=locus_dim,
            time_embed_dim=time_embed_dim,
            stick_latent_dim=latent_dim,
        ),
        ffn_cfg=dict(
            latent_dim=latent_dim,
            ffn_dim=ff_size,
            dropout=dropout,
            time_embed_dim=time_embed_dim
        ),
        text_encoder=dict(
            pretrained_model='clip',
            latent_dim=text_latent_dim,
            num_layers=2,
            ff_size=2048,
            dropout=dropout,
            use_text_proj=False
        ),
        multistick_encoder=dict(
            stick_encoder = stick_set['stickman_encoder'],
            weight=stickman_encoder_path,
            d_model=feat_dim,
            out_dim=latent_dim,
            ),
        locus_encoder=dict(
            input_dim=4, 
            latent_dim=locus_dim
            ),
        scale_func_cfg=dict(
            coarse_scale=4.0,
            both_coef=0.78123,
            text_coef=0.39284,
            retr_coef=-0.12475
        )
    ),
    loss_recon=dict(type='MSELoss', loss_weight=1, reduction='none'),
    diffusion_train=dict(
        beta_scheduler='linear',
        diffusion_steps=1000,
        model_mean_type='start_x',
        model_var_type='fixed_large',
    ),
    diffusion_test=dict(
        beta_scheduler='linear',
        diffusion_steps=1000,
        model_mean_type='start_x',
        model_var_type='fixed_large',
        respace='15,15,8,6,6',
    ),
    inference_type='ddim'
)
