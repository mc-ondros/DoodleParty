"""Unit tests for model architectures."""

import pytest
import numpy as np
import tensorflow as tf
from keras import Model, layers

from src.core.models import (
    build_custom_cnn,
    build_transfer_learning_resnet50,
    build_transfer_learning_mobilenetv3,
    build_transfer_learning_efficientnet,
    get_model,
    ARCHITECTURES
)


class TestCustomCNN:
    """Test custom CNN architecture."""
    
    def test_build_custom_cnn_default(self):
        """Test building custom CNN with default parameters."""
        model = build_custom_cnn()
        
        assert isinstance(model, Model)
        assert model.input_shape == (None, 28, 28, 1)
        # Model is built, check output shape through forward pass
        dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
        output = model(dummy_input)
        assert output.shape == (1, 1)
    
    def test_build_custom_cnn_custom_input_shape(self):
        """Test custom CNN with different input shape."""
        model = build_custom_cnn(input_shape=(128, 128, 1))

        assert model.input_shape == (None, 128, 128, 1)
        # Model is built, check output shape through forward pass
        # Use correct input shape that matches the model's input_shape
        dummy_input = np.random.rand(1, 128, 128, 1).astype(np.float32)
        output = model(dummy_input)
        assert output.shape == (1, 1)
    
    def test_build_custom_cnn_has_required_layers(self):
        """Test that custom CNN has expected layer types."""
        model = build_custom_cnn()
        
        layer_types = [type(layer).__name__ for layer in model.layers]
        
        # Should have Conv2D layers
        assert 'Conv2D' in layer_types
        # Should have BatchNormalization
        assert 'BatchNormalization' in layer_types
        # Should have MaxPooling2D
        assert 'MaxPooling2D' in layer_types
        # Should have Dropout
        assert 'Dropout' in layer_types
        # Should have Dense layers
        assert 'Dense' in layer_types
        # Should have Flatten
        assert 'Flatten' in layer_types
    
    def test_build_custom_cnn_trainable(self):
        """Test that custom CNN is trainable."""
        model = build_custom_cnn()
        
        assert model.trainable
        assert len(model.trainable_weights) > 0
    
    def test_build_custom_cnn_forward_pass(self):
        """Test forward pass through custom CNN."""
        model = build_custom_cnn()
        
        # Create dummy input
        X = np.random.rand(4, 28, 28, 1).astype(np.float32)
        
        # Forward pass
        predictions = model.predict(X, verbose=0)
        
        assert predictions.shape == (4, 1)
        # Sigmoid output should be in [0, 1]
        assert predictions.min() >= 0.0
        assert predictions.max() <= 1.0


class TestTransferLearningModels:
    """Test transfer learning model architectures."""
    
    def test_build_resnet50_default(self):
        """Test building ResNet50 with default parameters."""
        model, base_model = build_transfer_learning_resnet50()

        assert isinstance(model, Model)
        assert isinstance(base_model, Model)
        # Model is built, check output shape through forward pass
        import numpy as np
        dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
        output = model(dummy_input)
        assert output.shape == (1, 1)
    
    def test_build_resnet50_freeze_base(self):
        """Test that base model is frozen when specified."""
        model, base_model = build_transfer_learning_resnet50(freeze_base=True)
        
        assert not base_model.trainable
    
    def test_build_resnet50_unfreeze_base(self):
        """Test that base model is trainable when specified."""
        model, base_model = build_transfer_learning_resnet50(freeze_base=False)
        
        assert base_model.trainable
    
    def test_build_resnet50_forward_pass(self):
        """Test forward pass through ResNet50."""
        model, _ = build_transfer_learning_resnet50()
        
        # Create dummy input (28x28 will be resized to 224x224)
        X = np.random.rand(2, 28, 28, 1).astype(np.float32)
        
        predictions = model.predict(X, verbose=0)
        
        assert predictions.shape == (2, 1)
        assert predictions.min() >= 0.0
        assert predictions.max() <= 1.0
    
    def test_build_mobilenetv3_default(self):
        """Test building MobileNetV3 with default parameters."""
        model, base_model = build_transfer_learning_mobilenetv3()
        
        assert isinstance(model, Model)
        assert isinstance(base_model, Model)
        # Model is built, check output shape through forward pass
        import numpy as np
        dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
        output = model(dummy_input)
        assert output.shape == (1, 1)
    
    def test_build_mobilenetv3_freeze_base(self):
        """Test that MobileNetV3 base is frozen when specified."""
        model, base_model = build_transfer_learning_mobilenetv3(freeze_base=True)
        
        assert not base_model.trainable
    
    def test_build_mobilenetv3_forward_pass(self):
        """Test forward pass through MobileNetV3."""
        model, _ = build_transfer_learning_mobilenetv3()
        
        X = np.random.rand(2, 28, 28, 1).astype(np.float32)
        predictions = model.predict(X, verbose=0)
        
        assert predictions.shape == (2, 1)
        assert predictions.min() >= 0.0
        assert predictions.max() <= 1.0
    
    def test_build_efficientnet_default(self):
        """Test building EfficientNet with default parameters."""
        model, base_model = build_transfer_learning_efficientnet()
        
        assert isinstance(model, Model)
        assert isinstance(base_model, Model)
        # Model is built, check output shape through forward pass
        import numpy as np
        dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
        output = model(dummy_input)
        assert output.shape == (1, 1)
    
    def test_build_efficientnet_freeze_base(self):
        """Test that EfficientNet base is frozen when specified."""
        model, base_model = build_transfer_learning_efficientnet(freeze_base=True)
        
        assert not base_model.trainable
    
    def test_build_efficientnet_forward_pass(self):
        """Test forward pass through EfficientNet."""
        model, _ = build_transfer_learning_efficientnet()
        
        X = np.random.rand(2, 28, 28, 1).astype(np.float32)
        predictions = model.predict(X, verbose=0)
        
        assert predictions.shape == (2, 1)
        assert predictions.min() >= 0.0
        assert predictions.max() <= 1.0


class TestGetModel:
    """Test get_model function."""
    
    def test_get_model_custom(self, capsys):
        """Test getting custom architecture."""
        model, base_model = get_model('custom', summary=False)
        
        assert isinstance(model, Model)
        assert base_model is None
    
    def test_get_model_resnet50(self, capsys):
        """Test getting ResNet50 architecture."""
        model, base_model = get_model('resnet50', summary=False)
        
        assert isinstance(model, Model)
        assert isinstance(base_model, Model)
    
    def test_get_model_mobilenetv3(self, capsys):
        """Test getting MobileNetV3 architecture."""
        model, base_model = get_model('mobilenetv3', summary=False)
        
        assert isinstance(model, Model)
        assert isinstance(base_model, Model)
    
    def test_get_model_efficientnet(self, capsys):
        """Test getting EfficientNet architecture."""
        model, base_model = get_model('efficientnet', summary=False)
        
        assert isinstance(model, Model)
        assert isinstance(base_model, Model)
    
    def test_get_model_case_insensitive(self):
        """Test that architecture names are case-insensitive."""
        model1, _ = get_model('CUSTOM', summary=False)
        model2, _ = get_model('custom', summary=False)
        model3, _ = get_model('Custom', summary=False)
        
        # All should work
        assert isinstance(model1, Model)
        assert isinstance(model2, Model)
        assert isinstance(model3, Model)
    
    def test_get_model_invalid_architecture(self):
        """Test error handling for invalid architecture."""
        with pytest.raises(ValueError, match = 'Unknown architecture'):
            get_model('invalid_arch', summary=False)
    
    def test_get_model_freeze_base_parameter(self):
        """Test freeze_base parameter."""
        model1, base1 = get_model('resnet50', freeze_base=True, summary=False)
        model2, base2 = get_model('resnet50', freeze_base=False, summary=False)
        
        assert not base1.trainable
        assert base2.trainable
    
    def test_get_model_summary_output(self, capsys):
        """Test that summary is printed when requested."""
        model, _ = get_model('custom', summary=True)
        
        captured = capsys.readouterr()
        assert "Model Summary:" in captured.out
        assert "Total params" in captured.out or "Trainable params" in captured.out


class TestArchitectureComparison:
    """Test architecture comparison utilities."""
    
    def test_architectures_dict_exists(self):
        """Test that ARCHITECTURES dictionary is defined."""
        assert isinstance(ARCHITECTURES, dict)
        assert len(ARCHITECTURES) > 0
    
    def test_architectures_has_required_keys(self):
        """Test that all architectures have required metadata."""
        required_keys = ['params', 'size', 'speed', 'accuracy', 'pros', 'cons']
        
        for arch_name, arch_info in ARCHITECTURES.items():
            for key in required_keys:
                assert key in arch_info, f"{arch_name} missing {key}"
    
    def test_architectures_includes_all_models(self):
        """Test that all model types are in ARCHITECTURES."""
        expected_archs = ['custom', 'resnet50', 'mobilenetv3', 'efficientnet']
        
        for arch in expected_archs:
            assert arch in ARCHITECTURES


class TestModelProperties:
    """Test model properties and characteristics."""
    
    def test_models_have_sigmoid_output(self):
        """Test that all models use sigmoid activation for binary classification."""
        architectures = ['custom', 'resnet50', 'mobilenetv3', 'efficientnet']
        
        for arch in architectures:
            model, _ = get_model(arch, summary=False)
            
            # Last layer should have sigmoid activation
            last_layer = model.layers[-1]
            assert isinstance(last_layer, layers.Dense)
            assert last_layer.units == 1
            # Check activation (may be string or function)
            activation = last_layer.activation
            if hasattr(activation, '__name__'):
                assert activation.__name__ == 'sigmoid'
    
    def test_models_accept_grayscale_input(self):
        """Test that all models can process grayscale images."""
        architectures = ['custom', 'resnet50', 'mobilenetv3', 'efficientnet']
        X = np.random.rand(1, 28, 28, 1).astype(np.float32)
        
        for arch in architectures:
            model, _ = get_model(arch, summary=False)
            
            # Should not raise error
            predictions = model.predict(X, verbose=0)
            assert predictions.shape == (1, 1)
    
    def test_models_output_range(self):
        """Test that all models output probabilities in [0, 1]."""
        architectures = ['custom', 'resnet50', 'mobilenetv3', 'efficientnet']
        X = np.random.rand(10, 28, 28, 1).astype(np.float32)
        
        for arch in architectures:
            model, _ = get_model(arch, summary=False)
            predictions = model.predict(X, verbose=0)
            
            assert predictions.min() >= 0.0
            assert predictions.max() <= 1.0
