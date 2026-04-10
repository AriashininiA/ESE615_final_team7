from setuptools import setup, find_packages

setup(
    name="f1tenth_rl",
    version="1.0.0",
    description="Modular RL training framework for F1TENTH autonomous racing",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "PyYAML>=6.0",
        "gymnasium>=0.28.0",
        "torch>=2.0.0",
        "stable-baselines3[extra]>=2.1.0",
        "tensorboard>=2.14.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "Pillow>=9.5.0",
        "scikit-image>=0.21.0",
        "opencv-python>=4.8.0",
    ],
    extras_require={
        "wandb": ["wandb>=0.15.0"],
        "onnx": ["onnx>=1.14.0", "onnxruntime>=1.15.0"],
    },
    entry_points={
        "console_scripts": [
            "f1tenth-train=scripts.train:main",
            "f1tenth-eval=scripts.evaluate:main",
        ],
    },
)
