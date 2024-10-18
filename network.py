class GaussianActorCriticNet_SS_Comp(nn.Module, BaseNet):
    LOG_STD_MIN = -0.6931  # -20.
    LOG_STD_MAX = 0.4055  # 1.3
    FIXED_LOG_STD = -0.5  # Set your fixed log standard deviation value

    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random',
                 seed=1):
        super(GaussianActorCriticNet_SS_Comp, self).__init__()
        discrete_mask = False
        self.network = ActorCriticNetSSComp(state_dim, action_dim, phi_body, actor_body, critic_body,
                                             num_tasks, new_task_mask, discrete_mask=discrete_mask, seed=seed)
        self.task_label_dim = task_label_dim
        
        # Remove the fc_log_std layer since we are fixing log std
        # self.network.fc_log_std = CompBLC_MultitaskMaskLinear(...)

        self.network.actor_params += [p for p in self.network.fc_log_std.parameters() if p.requires_grad is True]
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False, to_numpy=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out
        mean = self.network.fc_action(phi_a)

        if to_numpy:
            return mean.cpu().detach().numpy()

        v = self.network.fc_critic(phi_v)

        # Set fixed log standard deviation instead of computing from fc_log_std
        log_std = self.FIXED_LOG_STD * torch.ones_like(mean)  # Assuming mean shape matches action shape
        log_std = torch.clamp(log_std, GaussianActorCriticNet_SS_Comp.LOG_STD_MIN,
                               GaussianActorCriticNet_SS_Comp.LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_std', std),
                              ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy()
        entropy = entropy.sum(-1).unsqueeze(-1)
        return mean, action, log_prob, entropy, v, layers_output