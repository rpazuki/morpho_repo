from enum import Enum
from itertools import cycle, zip_longest
import os
import pathlib
import pickle
from re import A
import warnings
from collections.abc import Iterable
import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
from .utils import create_dataset, lower_upper_bounds


def clip_by_value(z):
    return tf.clip_by_value(z, 1e-6, 1e10)


def clip_by_value_zero_lb(z):
    return tf.clip_by_value(z, 0, 1e10)


class Parameter_Type(Enum):
    CONSTANT = 1
    VARIABLE = 2
    INPUT = 3


class Loss_Grad_Type(Enum):
    BOTH = 1
    PARAMETER = 2
    PINN = 3


class PDE_Parameter:
    def __init__(
        self,
        name,
        parameter_type: Parameter_Type,
        value=1.0,
        index=-1,
        dtype=tf.float32,
        clip_constraint=clip_by_value,
    ):
        self.name = name
        self.parameter_type = parameter_type
        self.dtype = dtype
        self.value = value
        self.index = index
        self.clip_constraint = clip_constraint
        self.trainable = ()

    def build(self):
        if self.parameter_type == Parameter_Type.CONSTANT:
            self.tf_var = tf.constant(self.value, dtype=self.dtype, name=self.name)
        elif self.parameter_type == Parameter_Type.VARIABLE:
            self.tf_var = tf.Variable([self.value], dtype=self.dtype, name=self.name, constraint=self.clip_constraint)
            self.trainable = (self.tf_var,)
        else:  # INPUT
            self.tf_var = None

        return self

    def get_value(self, input):
        if self.parameter_type == Parameter_Type.CONSTANT:
            return self.tf_var
        elif self.parameter_type == Parameter_Type.VARIABLE:
            return self.tf_var
        else:  # INPUT, self.index + 3 starts after (x, y, t)
            return input[:, self.index + 3]

    def set_value(self, value):
        if self.parameter_type == Parameter_Type.CONSTANT:
            self.value = value
            try:
                self.tf_var.assign(value)
            except AttributeError:
                self.tf_var = tf.constant(self.value, dtype=self.dtype, name=self.name)
        elif self.parameter_type == Parameter_Type.VARIABLE:
            self.value = value
            try:
                self.tf_var[0].assign(value)
            except AttributeError:
                self.tf_var = tf.Variable([self.value], dtype=self.dtype, name=self.name, constraint=clip_by_value)
        # else:  do nothing


class TINN_Dataset(tf.data.Dataset):
    # def
    def __new__(cls, dtype, X, *args):

        tuple_of_list = (X,) + args
        lens = np.array([len(item) for item in tuple_of_list])
        max_index = np.where(lens == np.max(lens))[0][0]

        def gen_list():
            z = zip(
                *[tuple_of_list[i] if i == max_index else cycle(tuple_of_list[i]) for i in range(len(tuple_of_list))]
            )

            for items in z:
                yield items

            # if X_PDE is None:
            #     ds = tf.data.Dataset.from_tensor_slices((X, Y)).map(
            # lambda x, y: (tf.cast(x, dtype), tf.cast(y, dtype)))
            # else:

            #     def gen():
            #         l_x = len(X)
            #         l_pde = len(X_PDE)
            #         if l_x <= l_pde:
            #             for x, y, p in zip(cycle(X), cycle(Y), X_PDE):
            #                 yield (x, y, p)
            #         else:
            #             for x, y, p in zip(X, Y, cycle(X_PDE)):
            #                 yield (x, y, p)

        ds = tf.data.Dataset.from_generator(
            gen_list,
            output_types=tuple([dtype for _ in tuple_of_list]),
        )

        setattr(ds, "__parameters__", {})
        ds.__parameters__["sizes"] = lens
        setattr(ds, "sizes", ds.__parameters__["sizes"])
        ds.__parameters__["max_index"] = lens
        setattr(ds, "max_index", ds.__parameters__["max_index"])
        # ds.__parameters__["x_size"] = len(X)
        # setattr(ds, "x_size", ds.__parameters__["x_size"])
        # ds.__parameters__["x_pde_size"] = ds.x_size if X_PDE is None else len(X_PDE)
        # setattr(ds, "x_pde_size", ds.__parameters__["x_pde_size"])
        # ds.__parameters__["has_x_pde"] = False if X_PDE is None else True
        # setattr(ds, "has_x_pde", ds.__parameters__["has_x_pde"])

        def override_save(path_dir, name):
            return cls.save(ds, path_dir, name)

        setattr(ds, "save", override_save)
        # setattr(ds, "X", X)
        return ds

    def save(self, path_dir, name):
        path = pathlib.PurePath(path_dir).joinpath(name)
        with open(f"{str(path)}_data.pkl", "wb") as f:
            pickle.dump([item for item in self.as_numpy_iterator()], f)
        # if self.has_x_pde:
        #     with open(f"{str(path)}_X.pkl", "wb") as f:
        #         pickle.dump([(x.tolist()) for x, _, _ in self.as_numpy_iterator()], f)
        #     with open(f"{str(path)}_Y.pkl", "wb") as f:
        #         pickle.dump([(y.tolist()) for _, y, _ in self.as_numpy_iterator()], f)
        #     with open(f"{str(path)}_X_PDE.pkl", "wb") as f:
        #         pickle.dump([(x.tolist()) for _, _, x in self.as_numpy_iterator()], f)
        # else:
        #     with open(f"{str(path)}_X.pkl", "wb") as f:
        #         pickle.dump([(x.tolist()) for x, _ in self.as_numpy_iterator()], f)
        #     with open(f"{str(path)}_Y.pkl", "wb") as f:
        #         pickle.dump([(y.tolist()) for _, y in self.as_numpy_iterator()], f)

        with open(f"{str(path)}_parameters.pkl", "wb") as f:
            pickle.dump(self.__parameters__, f)

    @classmethod
    def restore(cls, path_dir, name, dtype=tf.float64):
        path = pathlib.PurePath(path_dir).joinpath(name)
        # asset = tf.saved_model.load(str(path))
        # with open(f"{str(path)}_X.pkl", "rb") as f:
        #     X = pickle.load(f)
        # with open(f"{str(path)}_Y.pkl", "rb") as f:
        #     Y = pickle.load(f)
        # if os.path.exists(f"{str(path)}_X_PDE.pkl"):
        #     with open(f"{str(path)}_X_PDE.pkl", "rb") as f:
        #         X_PDE = pickle.load(f)
        # else:
        #     X_PDE = None
        with open(f"{str(path)}_data.pkl", "rb") as f:
            items = pickle.load(f)
        with open(f"{str(path)}_parameters.pkl", "rb") as f:
            __parameters__ = pickle.load(f)

        ret = TINN_Dataset(dtype, *items)
        ret.__parameters__ = __parameters__
        for k, v in __parameters__.items():
            setattr(ret, k, v)

        return ret


class TINN_Single_Sim_Dataset(TINN_Dataset):
    def __new__(
        cls,
        path,
        name,
        dtype=tf.float64,
        thining_start=0,
        thining_step=0,
        pde_ratio=0,
        signal_to_noise=0,
        _X_internal_=None,
        __values__=None,
        __internal__=False,
    ):
        if __internal__:
            if __values__ is None:
                return super().__new__(cls, dtype, _X_internal_)
            else:
                return super().__new__(cls, dtype, _X_internal_, *__values__)

        data_path = pathlib.PurePath(path)
        with open(data_path.joinpath(f"{name}.npy"), "rb") as f:
            data = np.load(f)
        with open(data_path.joinpath("simulation.txt"), "r") as f:
            simulation = eval(f.read())
        t_star = np.linspace(simulation.t_start, simulation.t_end, simulation.t_steps)
        # Thining the dataset
        t_star = t_star[thining_start:]
        data = data[..., thining_start:]
        if thining_step > 0:
            t_star = t_star[::thining_step]
            data = data[..., ::thining_step]

        T = t_star.shape[0]

        L = simulation.L[0]
        x_size = simulation.n[0]  #
        y_size = simulation.n[1]  #
        N = x_size * y_size

        model_params = {"training_data_size": T * N, "signal_to_noise": signal_to_noise}
        if pde_ratio > 0:
            model_params = {**model_params, **{"pde_data_size": (T * N) / pde_ratio}}

        dataset = create_dataset(data, t_star, N, T, L, **model_params)
        obs_X = np.concatenate([dataset["obs_input"], dataset["obs_output"]], axis=1)
        if pde_ratio > 0:
            ds = super().__new__(cls, dtype, obs_X, dataset["pde"])
        else:
            ds = super().__new__(cls, dtype, obs_X)
        ds.__parameters__["lb"] = dataset["lb"]
        setattr(ds, "lb", ds.__parameters__["lb"])
        ds.__parameters__["ub"] = dataset["ub"]
        setattr(ds, "ub", ds.__parameters__["ub"])
        ds.__parameters__["simulation"] = simulation
        setattr(ds, "simulation", ds.__parameters__["simulation"])
        ds.__parameters__["ts"] = t_star
        setattr(ds, "ts", ds.__parameters__["ts"])
        # old_cache_method = ds.cache
        # old_batch_method = ds.batch

        # def set_att_from(ds_new, ds_old):
        #     setattr(ds_new, "x_size", ds_old.x_size)
        #     setattr(ds_new, "x_pde_size", ds_old.x_pde_size)
        #     setattr(ds_new, "lb", ds_old.lb)
        #     setattr(ds_new, "ub", ds_old.ub)
        #     setattr(ds_new, "simulation", ds_old.simulation)
        #     setattr(ds_new, "ts", ds_old.ts)
        #     # setattr(ds_new, "cache", ds_old.cache)

        # # setattr(ds_new, "batch", ds_old.batch)

        # def overide_cache(filename=""):
        #     ds2 = old_cache_method(filename)
        #     set_att_from(ds2, ds)
        #     return ds2

        # # def overide_batch(batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None, name=None):
        # #    ds2 = old_batch_method(batch_size, drop_remainder, num_parallel_calls, deterministic, name)
        # #    set_att_from(ds2, ds)
        # #    return ds2

        # setattr(ds, "cache", overide_cache)
        # setattr(ds, "batch", overide_batch)

        return ds

    @classmethod
    def restore(cls, path_dir, name, dtype=tf.float64):
        path = pathlib.PurePath(path_dir).joinpath(name)
        # asset = tf.saved_model.load(str(path))
        with open(f"{str(path)}_X.pkl", "rb") as f:
            X = pickle.load(f)
        # with open(f"{str(path)}_Y.pkl", "rb") as f:
        #    Y = pickle.load(f)
        if os.path.exists(f"{str(path)}_X_PDE.pkl"):
            with open(f"{str(path)}_X_PDE.pkl", "rb") as f:
                X_PDE = pickle.load(f)
                ret = TINN_Single_Sim_Dataset(None, None, _X_internal_=X, __values__=[X_PDE], __internal__=True)
        else:
            ret = TINN_Single_Sim_Dataset(None, None, _X_internal_=X, __internal__=True)
        with open(f"{str(path)}_parameters.pkl", "rb") as f:
            __parameters__ = pickle.load(f)

        for k, v in __parameters__.items():
            setattr(ret, k, v)
        ret.__parameters__ = __parameters__

        # lb, ub, simulation, ts = __parameters__
        # ret.__parameters__ = {"lb": lb, "ub": ub, "simulation": simulation, "ts": ts}
        # setattr(ret, "lb", ret.__parameters__["lb"])
        # setattr(ret, "ub", ret.__parameters__["ub"])
        # setattr(ret, "simulation", ret.__parameters__["simulation"])
        # setattr(ret, "ts", ret.__parameters__["ts"])

        return ret
        # path = pathlib.PurePath(path_dir).joinpath(name)
        # asset = tf.saved_model.load(str(path))
        # with open(f"{str(path)}_X.pkl", "rb") as f:
        #     X = pickle.load(f)
        # with open(f"{str(path)}_Y.pkl", "rb") as f:
        #     Y = pickle.load(f)
        # ds = TINN_Dataset(X, Y, shuffle=False, dtype=dtype)
        # old_cache_method = ds.cache

        # with open(f"{str(path)}_parameters.pkl", "rb") as f:
        #     lb, ub, simulation, ts = pickle.load(f)

        # setattr(ds, "lb", lb)
        # setattr(ds, "ub", ub)
        # setattr(ds, "simulation", simulation)
        # setattr(ds, "ts", ts)

        # def set_att_from(ds_new, ds_old):
        #     setattr(ds_new, "x_size", ds_old.x_size)
        #     setattr(ds_new, "x_pde_size", ds_old.x_pde_size)
        #     setattr(ds_new, "lb", ds_old.lb)
        #     setattr(ds_new, "ub", ds_old.ub)
        #     setattr(ds_new, "simulation", ds_old.simulation)
        #     setattr(ds_new, "ts", ds_old.ts)

        # def overide_cache(filename=""):
        #     ds2 = old_cache_method(filename)
        #     set_att_from(ds2, ds)
        #     return ds2

        # setattr(ds, "cache", overide_cache)
        # return ds


class TINN_Multiple_Sim_Dataset(TINN_Dataset):
    def __new__(
        cls,
        path,
        names,
        param_names,
        dtype=tf.float64,
        thining_start=0,
        thining_step=0,
        obs_ratio=1,
        pde_ratio=0,
        shuffle=True,
    ):
        assert len(names) > 0
        data_path = pathlib.PurePath(path)
        data_sources = []
        simulations = []
        ls_ts = []
        ls_obs_X = []
        ls_obs_Y = []
        ls_pde_X = []
        ubs = []
        lbs = []
        for name in names:
            with open(data_path.joinpath(name).joinpath(f"{name}.npy"), "rb") as f:
                data = np.load(f)
                data_sources += [data]
            with open(data_path.joinpath(name).joinpath("simulation.txt"), "r") as f:
                simulation = eval(f.read())
                simulations += [simulation]
            t_star = np.linspace(simulation.t_start, simulation.t_end, simulation.t_steps)
            # Thining the dataset
            t_star = t_star[thining_start:]
            data = data[..., thining_start:]
            if thining_step > 0:
                t_star = t_star[::thining_step]
                data = data[..., ::thining_step]

            T = t_star.shape[0]

            L = simulation.L[0]
            x_size = simulation.n[0]  #
            y_size = simulation.n[1]  #
            N = x_size * y_size

            model_params = {"training_data_size": (T * N) // obs_ratio}
            if pde_ratio > 0:
                model_params = {**model_params, **{"pde_data_size": (T * N) // pde_ratio}}

            dataset = create_dataset(data, t_star, N, T, L, **model_params)
            obs_X = dataset["obs_input"]
            obs_Y = dataset["obs_output"]
            if pde_ratio > 0:
                pde_X = dataset["pde"]
            else:
                pde_X = None
            ls_ts += [t_star]

            for k in param_names:
                obs_X = np.concatenate(
                    [obs_X, np.repeat(simulation.parameters[k], obs_X.shape[0])[..., np.newaxis]], axis=1
                )
                if pde_X is not None:
                    pde_X = np.concatenate(
                        [pde_X, np.repeat(simulation.parameters[k], pde_X.shape[0])[..., np.newaxis]], axis=1
                    )

            ls_obs_X += [obs_X]
            ls_obs_Y += [obs_Y]
            ls_pde_X += [pde_X]
            if pde_X is None:
                lb, ub = lower_upper_bounds(obs_X)
            else:
                lb, ub = lower_upper_bounds(np.concatenate([obs_X, pde_X], axis=0))

            for i, k in enumerate(param_names):
                # the first three columns are x, y, t. So, the index starts from 3.
                lb_i = lb[3 + i]
                ub_i = ub[3 + i]
                if lb_i == ub_i:
                    warnings.warn(f"Warning: the parameter '{k}' is constant: {lb_i}.")
                    # setting lb_1 = -1, lb_i = 1
                    lb[3 + i] = -1
                    ub[3 + i] = 1
            lbs += [lb]
            ubs += [ub]

        ds = super().__new__(
            cls,
            np.concatenate(ls_obs_X, axis=0),
            np.concatenate(ls_obs_Y, axis=0),
            None if ls_pde_X[0] is None else np.concatenate(ls_pde_X, axis=0),
            shuffle,
            dtype,
        )
        ds.__parameters__["lb"] = np.min(lbs, axis=0)
        setattr(ds, "lb", ds.__parameters__["lb"])
        ds.__parameters__["ub"] = np.max(ubs, axis=0)
        setattr(ds, "ub", ds.__parameters__["ub"])
        ds.__parameters__["simulations"] = simulations
        setattr(ds, "simulations", ds.__parameters__["simulations"])
        ds.__parameters__["ls_ts"] = ls_ts
        setattr(ds, "ls_ts", ds.__parameters__["ls_ts"])

        old_cache_method = ds.cache

        def set_att_from(ds_new, ds_old):
            setattr(ds_new, "x_size", ds_old.x_size)
            setattr(ds_new, "x_pde_size", ds_old.x_pde_size)
            setattr(ds_new, "lb", ds_old.lb)
            setattr(ds_new, "ub", ds_old.ub)
            setattr(ds_new, "simulations", ds_old.simulations)
            setattr(ds_new, "ls_ts", ds_old.ls_ts)

        def overide_cache(filename=""):
            ds2 = old_cache_method(filename)
            set_att_from(ds2, ds)
            return ds2

        setattr(ds, "cache", overide_cache)
        return ds


def minimize_parameters(
    pde_loss, pinn, inputs, parameters, norm=lambda x: np.sum(x**2), method="Powell", tol=1e-7, **kwargs
):

    # key_vals = [(v, v.tf_var.numpy()) for _, v in pde_loss.__dict__.items() if isinstance(v, PDE_Parameter)]
    key_vals = [(v, v.tf_var.numpy()) for v in parameters]
    initial_parameters = [(pde_param, v[0] if isinstance(v, Iterable) else v) for pde_param, v in key_vals]
    pde_params = [key for key, _ in initial_parameters]
    initial_tuple = tuple([v for _, v in initial_parameters])

    def minimize_model_parameters(args):
        for pde_param, value in zip(pde_params, args):
            pde_param.set_value(value)

        test_pde_u, test_pde_v = pde_loss.residual(pinn, inputs)
        return norm(test_pde_u) + norm(test_pde_v)

    return minimize(minimize_model_parameters, initial_tuple, method=method, tol=tol, **kwargs)
