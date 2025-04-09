#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin

from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ateacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # 设置配置参数
    cfg = setup(args)
    # 根据配置选择不同的训练器
    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        # 如果配置中未找到训练器名称，抛出异常
        raise ValueError("Trainer Name is not found.")

    # 如果只进行评估
    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ateacher":
            # 构建模型
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            # 构建集成模型
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            # 从检查点恢复或加载模型权重
            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            # 进行测试并返回结果
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            # 构建模型
            model = Trainer.build_model(cfg)
            # 从检查点恢复或加载模型权重
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            # 进行测试并返回结果
            res = Trainer.test(cfg, model)
        return res

    # 创建训练器实例
    trainer = Trainer(cfg)
    # 从检查点恢复或加载训练状态
    trainer.resume_or_load(resume=args.resume)

    # 进行训练并返回结果
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
