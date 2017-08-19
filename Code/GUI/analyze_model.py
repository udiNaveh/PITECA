class AnalyzeTabModel:

    def __init__(self, analysis_session_name, predicted_files_str, actual_files_str=None):
        self.analysis_session_name = analysis_session_name
        self.predicted_files_str = predicted_files_str
        self.actual_files_str = actual_files_str