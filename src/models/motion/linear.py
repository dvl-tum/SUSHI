import torch

class LinearMotionModel:
    def __init__(self):
        pass

    def __call__(self, x_motion, x_last_pos, pred_length, linear_center_only, **kwargs):
        """Estimate the velocity from the given trajectory (`x_motion`), and forward propagate it for 
           `pred_length` steps, starting from `x_last_pos`
        """
        num_nodes = x_motion.shape[0]
        if num_nodes > 0:
            # Velocity estimation
            vels = x_motion[:, 1:] - x_motion[:, :-1]
            num_estimates = vels.shape[1] -  torch.isnan(vels).sum(1)
            assert ((num_estimates.min(-1)[0] == num_estimates.max(-1)[0])).all(), "Some coordinates are Nans, some others are not?"
            num_estimates = num_estimates.min(-1)[0]
            
            vels = torch.nan_to_num(vels, nan = 0.0)
            estimate_vel = vels.sum(dim = 1) / num_estimates[:, None]
            assert not torch.isnan(estimate_vel).any()

            if linear_center_only:
                estimate_vel[..., 2:]=0

            # Prediction
            timesteps = torch.arange(start= 1, end= pred_length + 1, device=x_motion.device)
            displace = torch.einsum('nd,t->ntd',estimate_vel,timesteps)
            preds = displace + x_last_pos[:,  None]

            return {'pred_pos': preds, 'estimate_vel': estimate_vel}
        
        else:

            return {'pred_pos': torch.empty((0, pred_length, 4), device=x_motion.device), 
                    'estimate_vel': torch.empty((0, 4), device=x_motion.device)}