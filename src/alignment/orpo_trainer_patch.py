import torch

from trl import ORPOTrainer


class ORPOTrainerPatch(ORPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def odds_ratio_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """Compute ORPO's odds ratio (OR) loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the ORPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The log odds ratio of the chosen responses over the rejected responses ratio for logging purposes.
            The `log(sigmoid(log_odds_chosen))` for logging purposes.
        """
        EPS = 1e-6
        MAX_LOGIT = torch.log(torch.tensor(1.0 - EPS))

        policy_chosen_logps = torch.clamp(policy_chosen_logps, min=-20, max=MAX_LOGIT)
        policy_rejected_logps = torch.clamp(
            policy_rejected_logps, min=-20, max=MAX_LOGIT
        )

        def log1mexp(x):
            # Ensure x <= 0 (valid for log probabilities of p in (0, 1])
            # For x < -0.693147 (i.e., exp(x) < 0.5), use log(-expm1(x))
            # For x >= -0.693147, use log1p(-exp(x))
            return torch.where(
                x < -0.693147, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x))
            )

        policy_chosen_logps_fl32 = policy_chosen_logps.to(torch.float32)
        policy_rejected_logps_fl32 = policy_rejected_logps.to(torch.float32)

        # Compute log(1 - exp(x)) stably
        log1mexp_pcl = log1mexp(policy_chosen_logps_fl32)
        log1mexp_prl = log1mexp(policy_rejected_logps_fl32)

        # Compute log_odds in float32
        log_odds = (policy_chosen_logps_fl32 - policy_rejected_logps_fl32) - (
            log1mexp_pcl - log1mexp_prl
        )

        # log_odds = (policy_chosen_logps - policy_rejected_logps) - (
        #     torch.log1p(-torch.exp(policy_chosen_logps))
        #     - torch.log1p(-torch.exp(policy_rejected_logps))
        # )

        ratio = torch.nn.functional.logsigmoid(log_odds)
        losses = self.beta * ratio

        chosen_rewards = (
            self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()
        )

        return (
            losses,
            chosen_rewards,
            rejected_rewards,
            torch.mean(ratio),
            torch.mean(log_odds),
        )
