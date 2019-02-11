import tensorflow as tf
from edward.models import Dirichlet
from GenerativeModel import joint_GBDS
from RecognitionModel import SmoothingPastLDSTimeSeries
# from RecognitionModel import SmoothingPastLDSTimeSeries, joint_recognition
# from utils import get_vel
# from tf_generate_trial import (recover_orig_val, recover_normalized,
#                                generate_weight, generate_rotation_mat,
#                                generate_prey_trajectory,
#                                generate_second_prey_trajectory,
#                                generate_predator_trajectory)


# q_g_init_ep = 10
# q_u_init_ep = 0

def get_GMM(NN, inputs, K, D, name="GMM"):
    with tf.name_scope(name):
        NN_output = tf.identity(NN(inputs), "NN_output")
        mu = tf.reshape(
            NN_output[:, :, :(K * D)],
            [tf.shape(inputs)[0], -1, K, D], "mu")
        lmbda = tf.reshape(tf.nn.softplus(
            NN_output[:, :, (K * D):(2 * K * D)], "softplus_lambda"),
            [tf.shape(inputs)[0], -1, K, D], "lambda")
        w = tf.nn.softmax(tf.reshape(
            NN_output[:, :, (2 * K * D):],
            [tf.shape(inputs)[0], -1, K], "reshape_w"), -1, "w")

        return mu, lmbda, w

class game_model(object):
    """Auxiliary class to construct the computational graph
    (define generative and recognition models, draw samples, trial completion)
    """
    def __init__(self, params, inputs, max_vel, state_dim, extra_dim=0,
                 n_samples=50):
        with tf.name_scope(params["name"]):
            model_dim = params["model_dim"]
            obs_dim = params["obs_dim"]
            latent_u = params["latent_u"]
            clip_range = params["clip_range"]

            traj = inputs["trajectory"]
            states = inputs["states"]
            ctrl_obs = inputs["ctrl_obs"]
            extra_conds = inputs["extra_conds"]
            # epoch = inputs["epoch"]

            self.var_list = []
            self.log_vars = []

            with tf.name_scope("variable_value_shape"):
                traj_shape = traj.shape.as_list()
                if traj_shape[0] is None:
                    B = 1
                else:
                    B = traj_shape[0]
                if traj_shape[1] is None:
                    Tt = 2
                else:
                    Tt = traj_shape[1]
                if latent_u:
                    value_shape = [B, Tt, model_dim * 2]
                else:
                    value_shape = [B, Tt, model_dim]

            # p_alpha_vars, _, self.p_alpha = get_alpha_gen(
            #     params["p_params"][0]["A_NN"], states, extra_conds, extra_dim)
            # self.var_list += p_alpha_vars
            # self.log_vars += p_alpha_vars
            # self.p_alpha = Alpha_gen(
            #     params["p_params"][0]["A_NN"], states, extra_conds, extra_dim)
            # self.var_list += self.p_alpha.vars
            # self.log_vars += self.p_alpha.vars

            # q_alpha_vars, q_alpha_params, self.q_alpha = get_alpha_rec(
            #     params["q_params"]["NN_Alpha"], traj, extra_conds, obs_dim,
            #     extra_dim, params["q_params"]["lag"])
            # self.var_list += q_alpha_vars
            # self.log_vars += q_alpha_vars
            # self.q_alpha = Alpha_rec(
            #     params["q_params"]["NN_Alpha"], traj, extra_conds, obs_dim,
            #     extra_dim, params["q_params"]["lag"])
            # self.var_list += self.q_alpha.vars
            # self.log_vars += self.q_alpha.vars

            self.p = joint_GBDS(
                params["p_params"], model_dim, states, ctrl_obs, extra_conds,
                latent_u, name="generative", value=tf.zeros(value_shape))
            # self.p = joint_GBDS(
            #     params["p_params"], model_dim, states, ctrl_obs, extra_conds,
            #     epoch, latent_u, name="generative",
            #     value=tf.zeros(value_shape))
            self.var_list += self.p.var_list
            self.log_vars += self.p.log_vars

            # q_g_trainable = tf.greater(epoch, q_g_init_ep, "q_g_trainable")
            # q_u_trainable = tf.greater(epoch, q_u_init_ep, "q_u_trainable")
            if latent_u:
                self.q = SmoothingPastLDSTimeSeries(
                    params["q_params"], traj, model_dim * 2, obs_dim,
                    extra_conds, name="recognition")
                # self.q = joint_recognition(
                #     params["q_params"]["q_g"], params["q_params"]["q_u"],
                #     traj, model_dim, obs_dim, extra_conds,
                #     q_g_trainable, q_u_trainable, name="recognition")
            else:
                self.q = SmoothingPastLDSTimeSeries(
                    params["q_params"], traj, model_dim, obs_dim,
                    extra_conds, name="recognition")
                # self.q = SmoothingPastLDSTimeSeries(
                #     params["q_params"], traj, model_dim, obs_dim,
                #     extra_conds, q_g_trainable, name="recognition")
            self.var_list += self.q.var_list
            self.log_vars += self.q.log_vars

            # self.latent_vars = {self.p: self.q}
            self.latent_vars = {self.p.agents[0].alpha: self.q.alpha,
                                self.p: self.q}

            with tf.name_scope("G"):
                # game state includes both position and velocity
                s = tf.placeholder(tf.float32, [None, None, state_dim],
                                   "state")
                npcs = tf.placeholder(
                    tf.float32, [None, None, extra_dim],
                    "extra_conditions")

                with tf.name_scope("subject"):
                    a = self.p.agents[0]
                    G0_mu, G0_lambda, G0_w = get_GMM(
                        a.G0_NN, s, a.K, a.dim, "G0")

                    # no_first_npc = tf.reduce_all(tf.equal(tf.gather(
                    #     npcs, [extra_dim // 2 - 1], axis=-1), 0),
                    #     name="first_npc_bool")
                    no_second_npc = tf.reduce_all(tf.equal(tf.gather(
                        npcs, [extra_dim - 1], axis=-1), 0),
                        name="second_npc_bool")

                    s1 = tf.concat([s, npcs[:, :, :(extra_dim // 2)]], -1,
                                   "s1")
                    G1_mu_1, G1_lambda_1, G1_w_1 = get_GMM(
                        a.G1_NN, s1, a.K, a.dim, "G1_1")

                    s2 = tf.concat([s, npcs[:, :, (extra_dim // 2):]], -1,
                                   "s2")
                    G1_mu_2, G1_lambda_2, G1_w_2 = get_GMM(
                        a.G1_NN, s2, a.K, a.dim, "G1_2")

                    # alpha = tf.identity(tf.case(
                    #     {no_first_npc: lambda: tf.nn.softmax(
                    #         tf.gather(a.A_NN(tf.concat([s, npcs], -1)),
                    #                   [0, 2], axis=-1), -1),
                    #      no_second_npc: lambda: tf.nn.softmax(
                    #         tf.gather(a.A_NN(tf.concat([s, npcs], -1)),
                    #                   [0, 1], axis=-1), -1)},
                    #     default=lambda: tf.nn.softmax(
                    #         a.A_NN(tf.concat([s, npcs], -1)), -1),
                    #     exclusive=True), "alpha")
                    O1 = a.A_NN(tf.concat([s, npcs], -1))
                    O2 = a.A_NN(tf.concat([s, tf.concat(
                        [npcs[:, :, (extra_dim // 2):],
                         npcs[:, :, :(extra_dim // 2)]], -1)], -1))
                    A0 = (tf.gather(O1, [0], axis=-1) +
                          tf.gather(O2, [0], axis=-1)) / 2.
                    A1 = tf.gather(O1, [1], axis=-1)
                    A2 = tf.gather(O2, [1], axis=-1)
                    # A1 = (tf.gather(O1, [1], axis=-1) +
                    #       tf.gather(O2, [2], axis=-1)) / 2.
                    # A2 = (tf.gather(O1, [2], axis=-1) +
                    #       tf.gather(O2, [1], axis=-1)) / 2.

                    # alpha = tf.identity(tf.cond(
                    #     no_second_npc,
                    #     lambda: tf.nn.softmax(tf.concat([A0, A1], -1), -1),
                    #     lambda: tf.nn.softmax(tf.concat(
                    #         [A0, A1, A2], -1), -1)), "alpha")
                    p_alpha_concentration = tf.identity(tf.cond(
                        no_second_npc,
                        lambda: tf.nn.softmax(tf.concat([A0, A1], -1), -1),
                        lambda: tf.nn.softmax(tf.concat(
                            [A0, A1, A2], -1), -1)), "alpha_concentration")
                    p_alpha_sample = tf.identity(
                        Dirichlet(p_alpha_concentration).sample(), "alpha")

                    # G_mu = tf.identity(tf.case(
                    #     {no_first_npc: lambda: tf.concat(
                    #         [G0_mu, G1_mu_2], 2),
                    #      no_second_npc: lambda: tf.concat(
                    #         [G0_mu, G1_mu_1], 2)},
                    #     default=lambda: tf.concat(
                    #         [G0_mu, G1_mu_1, G1_mu_2], 2),
                    #     exclusive=True), "mu")
                    # G_lambda = tf.identity(tf.case(
                    #     {no_first_npc: lambda: tf.concat(
                    #         [G0_lambda, G1_lambda_2], 2),
                    #      no_second_npc: lambda: tf.concat(
                    #         [G0_lambda, G1_lambda_1], 2)},
                    #     default=lambda: tf.concat(
                    #         [G0_lambda, G1_lambda_1, G1_lambda_2], 2),
                    #     exclusive=True), "lambda")
                    # G_w = tf.identity(tf.case(
                    #     {no_first_npc: lambda: tf.concat(
                    #         [G0_w * tf.gather(alpha, [0], axis=-1),
                    #          G1_w_2 * tf.gather(alpha, [1], axis=-1)], -1),
                    #      no_second_npc: lambda: tf.concat(
                    #         [G0_w * tf.gather(alpha, [0], axis=-1),
                    #          G1_w_1 * tf.gather(alpha, [1], axis=-1)], -1)},
                    #     default=lambda: tf.concat(
                    #         [G0_w * tf.gather(alpha, [0], axis=-1),
                    #          G1_w_1 * tf.gather(alpha, [1], axis=-1),
                    #          G1_w_2 * tf.gather(alpha, [2], axis=-1)], -1),
                    #     exclusive=True), "w")
                    G_mu = tf.identity(tf.cond(
                        no_second_npc, lambda: tf.concat([G0_mu, G1_mu_1], 2),
                        lambda: tf.concat([G0_mu, G1_mu_1, G1_mu_2], 2)),
                        "mu")
                    G_lambda = tf.identity(tf.cond(
                        no_second_npc,
                        lambda: tf.concat([G0_lambda, G1_lambda_1], 2),
                        lambda: tf.concat(
                            [G0_lambda, G1_lambda_1, G1_lambda_2], 2)),
                        "lambda")
                    G_w = tf.identity(tf.cond(
                        no_second_npc, lambda: tf.concat(
                            [G0_w * tf.gather(p_alpha_sample, [0], axis=-1),
                             G1_w_1 * tf.gather(
                                p_alpha_sample, [1], axis=-1)], -1),
                        lambda: tf.concat(
                            [G0_w * tf.gather(p_alpha_sample, [0], axis=-1),
                             G1_w_1 * tf.gather(p_alpha_sample, [1], axis=-1),
                             G1_w_2 * tf.gather(
                                p_alpha_sample, [2], axis=-1)], -1)), "w")

            with tf.name_scope("posterior"):
                with tf.name_scope("means"):
                    q_g_mu = tf.identity(
                        tf.squeeze(self.q.postX, -1)[:, :, :model_dim],
                        "goal")
                    if latent_u:
                        q_u_mu = tf.identity(
                            tf.squeeze(self.q.postX, -1)[:, :, model_dim:],
                            "control")
                with tf.name_scope("samples"):
                    q_samp = self.q.sample(n_samples)
                    q_g_samp = tf.identity(
                        q_samp[:, :, :, :model_dim], "goal")
                    if latent_u:
                        q_u_samp = tf.identity(
                            q_samp[:, :, :, model_dim:], "control")
                with tf.name_scope("alpha"):
                    # q_alpha_params = tf.identity(
                    #     q_alpha_params, "parameters")
                    # q_alpha_sample = tf.identity(
                    #     self.q_alpha.sample(), "sample")
                    q_alpha_concentration = tf.identity(
                        self.q.alpha.concentration, "concentration")
                    q_alpha_sample = tf.identity(
                        self.q.alpha.sample(), "sample")

            with tf.name_scope("update_one_step"):
                prev_y = tf.placeholder(
                    tf.float32, obs_dim, "previous_position")
                curr_y = tf.placeholder(
                    tf.float32, obs_dim, "current_position")
                v = tf.divide(curr_y - prev_y, max_vel,
                              "current_velocity")
                curr_s = tf.concat([curr_y, v], 0, "current_state")

                if inputs["extra_conds"] is not None:
                    gen_extra_conds = tf.placeholder(
                        tf.float32, extra_dim, "extra_conditions")
                else:
                    gen_extra_conds = None

                with tf.name_scope("goal"):
                    prev_g = tf.placeholder(
                        tf.float32, model_dim, "previous")
                    curr_g = tf.identity(self.p.update_goal(
                        curr_s, prev_g, gen_extra_conds), "current")

                with tf.name_scope("control"):
                    with tf.name_scope("error"):
                        curr_error = tf.subtract(
                            curr_g, curr_y, "current")
                        prev_error = tf.placeholder(
                            tf.float32, model_dim, "previous")
                        prev2_error = tf.placeholder(
                            tf.float32, model_dim, "previous2")
                        errors = tf.stack(
                            [prev2_error, prev_error, curr_error], 0,
                            "all")
                    prev_u = tf.placeholder(
                        tf.float32, model_dim, "previous")
                    curr_u = tf.identity(self.p.update_ctrl(
                        errors, prev_u), "current")

                if latent_u:
                    next_y = tf.clip_by_value(
                        curr_y + max_vel * tf.clip_by_value(
                            curr_u, clip_range[:, 0], clip_range[:, 1],
                            "clip_control"), -1., 1., "next_position")
                else:
                    next_y = tf.clip_by_value(
                        curr_y + max_vel * tf.tanh(curr_u), -1., 1.,
                        "next_position")
