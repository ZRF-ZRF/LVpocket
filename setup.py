import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LVPocket",
    version="0.0.1",
    author="CPU-409",
    author_email="3221051463@stu.cpu.edu.cn",
    keywords="protein binding pockets prediction, lvnet",
    description="A Protein Binding Pocket Prediction Methods.",
    long_description="We proposed LVPocket, a novel method that synergistically captures both local" \
                     " and global information of protein data through the integration of Transformer" \
                     " encoders, which help the model achieve better performance in binding pockets prediction. " \
                     "And then we tailored prediction models for data of four distinct structural classes of " \
                     "proteins using the transfer learning. The four fine-tuned models were trained on the baseline" \
                     " LVPocket model which was trained on the sc-PDB dataset. LVPocket exhibits superior performance" \
                     " on three independent datasets compared to current state-of-the-art methods. Additionally, the " \
                     "fine-tuned model outperforms the baseline model in terms of performance.",
    long_description_content_type="text/markdown",
    url="https://github.com/ZRF-ZRF/LVpocket.git",
    packages=setuptools.find_packages(),
    python_requires=">=3.6, <=3.7",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
    ],
)