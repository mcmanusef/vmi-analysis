import uconn_pipelines
from vmi_analysis.acquisition_ui import MainApp
from vmi_analysis.processing.pipelines import TPXFileConverter

if __name__ == "__main__":
    pipelines = {
        'TPX Converter': (TPXFileConverter, ".h5"),
        'CV4 Converter': (uconn_pipelines.CV4ConverterPipeline, ".cv4")
    }
    app = MainApp(
        processing_pipelines=pipelines,
        test_dir=r"C:\serval_test",
        default_dir=r"C:\DATA\%Y%m%d")
    app.mainloop()
