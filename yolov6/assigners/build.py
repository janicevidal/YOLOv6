from yolov6.assigners.align_ota_assigner import AlignOTAAssigner
from yolov6.assigners.dsl_assigner import DynamicSoftLabelAssigner as MMYOLODynamicSoftLabelAssigner
from yolov6.assigners.dynamic_soft_label_assigner import DynamicSoftLabelAssigner
from yolov6.assigners.stal_assigner import STALAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner


def build_assigner(assigner_type='tal', num_classes=80, tal_topk=13,
                   assigner_kwargs=None, fpn_strides=None):
    """Build the formal assigner; ATSS warmup is controlled by the loss."""
    kwargs = dict(assigner_kwargs or {})
    if assigner_type == 'tal':
        kwargs.setdefault('topk', tal_topk)
        kwargs.setdefault('alpha', 1.0)
        kwargs.setdefault('beta', 6.0)
        return TaskAlignedAssigner(num_classes=num_classes, **kwargs)
    if assigner_type == 'stal':
        kwargs.setdefault('topk', tal_topk)
        if fpn_strides is not None:
            kwargs.setdefault('stride', fpn_strides)
        return STALAssigner(num_classes=num_classes, **kwargs)
    if assigner_type == 'alignota':
        return AlignOTAAssigner(num_classes=num_classes, **kwargs)
    if assigner_type == 'dsl':
        return DynamicSoftLabelAssigner(num_classes=num_classes, **kwargs)
    if assigner_type in ('mmyolo_dsl', 'batch_dsl'):
        return MMYOLODynamicSoftLabelAssigner(num_classes=num_classes, **kwargs)
    raise ValueError("assigner_type must be 'tal', 'stal', 'alignota', 'dsl' or 'mmyolo_dsl' ('batch_dsl'), "
                     "got {!r}".format(assigner_type))
