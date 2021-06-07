import os

from typing import Dict, List, Tuple, Union

from util import decode_history_file, decode_debug_file, decode_log_file, decode_keras_model, decode_loss_history_plot


class OutputCollector:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.out_data = os.path.join(output_dir, "data")
        self.out_debug = os.path.join(output_dir, "debug")
        self.out_log = os.path.join(output_dir, "log")
        self.out_model = os.path.join(output_dir, "model")
        self.out_plot = os.path.join(output_dir, "plot")
        self.out_eval = os.path.join(output_dir, "eval")

    @staticmethod
    def encode_key(start: str, source_port_name: str = None, config_uid: str = None) -> str:
        if config_uid is None:
            config_uid = "-1"
        return f"{start}_{source_port_name}_{config_uid}" if config_uid != "-1" else f"{start}"

    @staticmethod
    def decode_key(key: str) -> Tuple[str, str, str]:
        if "_" in key:
            result = key.split("_")
            return result[0], result[1], result[2]
        else:
            return key, "", "-1"

    def collect_data(self, port_file_name: str, file_type: str, full_path: bool = True,
                     group: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        self._check_file_type("data", file_type)
        file_type = f"history_{file_type}"
        out_data = os.path.join(self.out_data, port_file_name)
        if not os.path.exists(out_data):
            return {} if group else []
        data_files = []
        groups = {}
        for file in filter(lambda f: f.startswith(file_type), os.listdir(out_data)):
            result = os.path.join(out_data, file) if full_path else file
            if group:
                _, _, _, start, source_port, config_uid = decode_history_file(file)
                key = self.encode_key(start, source_port, str(config_uid))
                if key in groups:
                    groups[key].append(result)
                else:
                    groups[key] = [result]
            else:
                data_files.append(result)
        return groups if group else data_files

    def collect_debug(self, port_file_name: str, file_type: str, full_path: bool = True,
                      group: bool = False) -> Union[List[str], Dict[str, Dict[int, List[str]]]]:
        self._check_file_type("debug", file_type)
        file_type = f"debug-{file_type}"
        out_debug = os.path.join(self.out_debug, port_file_name)
        if not os.path.exists(out_debug):
            return {} if group else []
        debug_files = []
        groups = {}
        for file in filter(lambda f: f.startswith(file_type), os.listdir(out_debug)):
            result = os.path.join(out_debug, file) if full_path else file
            if group:
                _, _, start, config_uid = decode_debug_file(file)
                key = self.encode_key(start, "", str(config_uid))
                if key in groups:
                    groups[key].append(result)
                else:
                    groups[key] = [result]
            else:
                debug_files.append(result)
        return groups if group else debug_files

    def collect_log(self, port_file_name: str, file_type: str, full_path: bool = True,
                    group: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        self._check_file_type("log", file_type)
        file_type = f"train-log-{file_type}"
        out_log = os.path.join(self.out_log, port_file_name)
        if not os.path.exists(out_log):
            return {} if group else []
        log_files = []
        groups = {}
        for file in filter(lambda f: f.startswith(file_type), os.listdir(out_log)):
            result = os.path.join(out_log, file) if full_path else file
            if group:
                _, _, start, config_uid = decode_log_file(file)
                key = self.encode_key(start, "", str(config_uid))
                if key in groups:
                    groups[key].append(result)
                else:
                    groups[key] = [result]
            else:
                log_files.append(result)
        return groups if group else log_files

    def collect_model(self, port_file_name: str, file_type: str, full_path: bool = True,
                      group: bool = False) -> Union[List[str], Dict[str, str]]:
        self._check_file_type("model", file_type)
        out_model = os.path.join(self.out_model, port_file_name)
        if not os.path.exists(out_model):
            return {} if group else []
        model_files = []
        groups = {}
        for file in filter(lambda f: f.endswith(".h5") and f.startswith(file_type), os.listdir(out_model)):
            result = os.path.join(out_model, file) if full_path else file
            if group:
                _, _, start, source_port, config_uid = decode_keras_model(file)
                key = self.encode_key(start, source_port, str(config_uid))
                if start in groups:
                    raise ValueError(f"Model collecting error: Key '{key}' is not unique in '{port_file_name}'")
                else:
                    groups[key] = result
            else:
                model_files.append(result)
        return groups if group else model_files

    def collect_plot(self, plot_file_name: str, file_type: str, full_path: bool = True,
                     group: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        self._check_file_type("plot", file_type)
        file_type = f"history_{file_type}"
        out_plot = os.path.join(self.out_plot, plot_file_name)
        if not os.path.exists(out_plot):
            return {} if group else []
        plot_files = []
        groups = {}
        for file in filter(lambda f: f.startswith(file_type), os.listdir(out_plot)):
            result = os.path.join(out_plot, file) if full_path else file
            if group:
                _, _, start, _, source_port, config_uid = decode_loss_history_plot(file)
                key = self.encode_key(start, source_port, str(config_uid))
                if key in groups:
                    groups[key].append(result)
                else:
                    groups[key] = [result]
            else:
                plot_files.append(result)
        return groups if group else plot_files

    def collect_eval(self, port_file_name: str, file_type: str, full_path: bool = True,
                     group: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        self._check_file_type("eval", file_type)
        file_type = f"loss-{file_type}"
        out_eval = os.path.join(self.out_eval, port_file_name)
        if not os.path.exists(out_eval):
            return {} if group else []
        plot_files = []
        groups = {}
        for file in filter(lambda f: f.startswith(file_type), os.listdir(out_eval)):
            result = os.path.join(out_eval, file) if full_path else file
            if group:
                _, _, start, _, source_port, config_uid = decode_loss_history_plot(file)
                key = self.encode_key(start, source_port, str(config_uid))
                if key in groups:
                    groups[key].append(result)
                else:
                    groups[key] = [result]
            else:
                plot_files.append(result)
        return groups if group else plot_files

    @staticmethod
    def _check_file_type(output_file: str, file_type: str) -> None:
        if file_type not in ["base", "transfer"]:
            raise ValueError(f"Unable to collect {output_file}: File type not in [base, transfer])")
