import os

from typing import Dict, List, Union

from util import decode_loss_file, decode_debug_file, decode_log_file, decode_model_file, decode_loss_plot


class OutputCollector:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.out_data = os.path.join(output_dir, "data")
        self.out_debug = os.path.join(output_dir, "debug")
        self.out_log = os.path.join(output_dir, "log")
        self.out_model = os.path.join(output_dir, "model")
        self.out_plot = os.path.join(output_dir, "plot")

    def collect_data(self, port_file_name: str, full_path: bool = True, group: bool = False) -> Union[List[str],
                                                                                                      Dict[str, str]]:
        out_data = os.path.join(self.out_data, port_file_name)
        data_files = []
        groups = {}
        for file in filter(lambda f: f.startswith("loss"), os.listdir(out_data)):
            result = os.path.join(out_data, file) if full_path else file
            if group:
                _, _, start = decode_loss_file(file)
                groups[start] = result
            else:
                data_files.append(result)
        return groups if group else data_files

    def collect_debug(self, port_file_name: str, full_path: bool = True, group: bool = False) -> Union[List[str],
                                                                                                       Dict[str, str]]:
        out_debug = os.path.join(self.out_debug, port_file_name)
        debug_files = []
        groups = {}
        for file in filter(lambda f: f.startswith("debug"), os.listdir(out_debug)):
            result = os.path.join(out_debug, file) if full_path else file
            if group:
                _, _, start = decode_debug_file(file)
                groups[start] = result
            else:
                debug_files.append(result)
        return groups if group else debug_files

    def collect_log(self, port_file_name: str, full_path: bool = True, group: bool = False) -> Union[List[str],
                                                                                                     Dict[str, str]]:
        out_log = os.path.join(self.out_log, port_file_name)
        log_files = []
        groups = {}
        for file in filter(lambda f: f.startswith("train-log"), os.listdir(out_log)):
            result = os.path.join(out_log, file) if full_path else file
            if group:
                _, _, start = decode_log_file(file)
                groups[start] = result
            else:
                log_files.append(result)
        return groups if group else log_files

    def collect_model(self, port_file_name: str, full_path: bool = True, group: bool = False) -> Union[List[str],
                                                                                                       Dict[str, str]]:
        out_model = os.path.join(self.out_model, port_file_name)
        model_files = []
        groups = {}
        for file in filter(lambda f: f.endswith(".pt") and f.startswith("model"), os.listdir(out_model)):
            result = os.path.join(out_model, file) if full_path else file
            if group:
                _, _, start, _ = decode_model_file(file)
                groups[start] = result
            else:
                model_files.append(result)
        return groups if group else model_files

    def collect_plot(self, port_file_name: str, full_path: bool = True,
                     group: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        out_plot = os.path.join(self.out_plot, port_file_name)
        plot_files = []
        groups = {}
        for file in filter(lambda f: f.startswith("loss"), os.listdir(out_plot)):
            result = os.path.join(out_plot, file) if full_path else file
            if group:
                _, _, start, _ = decode_loss_plot(file)
                if start in groups:
                    groups[start].append(result)
                else:
                    groups[start] = [result]
            else:
                plot_files.append(result)
        return groups if group else plot_files
