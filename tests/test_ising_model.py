"""
Tests for the Ising Model implementation.
"""

import numpy as np
import pytest
from ising_model.model import IsingModel
from ising_model.utils import create_domain_wall, create_checkerboard

class TestIsingModel:
    """Test cases for the IsingModel class."""
    
    def test_initialization(self):
        """Test model initialization with different parameters."""
        # Test default initialization
        model = IsingModel(size=10, temperature=2.0)
        assert model.size == 10
        assert model.temperature == 2.0
        assert model.lattice.shape == (10, 10)
        assert set(np.unique(model.lattice)) == {-1, 1}
        
        # Test with custom initial state
        initial_state = np.ones((10, 10))
        model = IsingModel(size=10, temperature=2.0, initial_state=initial_state)
        np.testing.assert_array_equal(model.lattice, initial_state)
    
    def test_energy_calculation(self):
        """Test energy calculation for known configurations."""
        # All spins aligned (minimum energy)
        size = 4
        initial_state = np.ones((size, size))
        model = IsingModel(size=size, temperature=1.0, initial_state=initial_state)
        
        # For all spins aligned, energy should be -2*N^2 (4 neighbors per spin, each pair counted twice)
        expected_energy = -2 * size**2
        assert model.energy == expected_energy
        
        # Check that flipping one spin changes energy as expected
        old_energy = model.energy
        dE = model._delta_energy(0, 0)  # Should be 8 (4 neighbors * 2)
        model.lattice[0, 0] *= -1
        new_energy = model._calculate_energy()
        assert new_energy == old_energy + dE
    
    def test_magnetization(self):
        """Test magnetization calculation."""
        size = 4
        initial_state = np.ones((size, size))
        model = IsingModel(size=size, temperature=1.0, initial_state=initial_state)
        
        # All spins up
        assert model.magnetization == size**2
        
        # Flip one spin
        model.lattice[0, 0] = -1
        assert model.magnetization == size**2 - 2  # Decreased by 2 (from +1 to -1)
    
    def test_simulation(self):
        """Test running a short simulation."""
        model = IsingModel(size=10, temperature=2.0)
        initial_energy = model.energy
        
        # Run a few steps
        energies, magnetizations = model.simulate(steps=10)
        
        # Check that we got the expected number of samples
        assert len(energies) == 10
        assert len(magnetizations) == 10
        
        # Energy should have changed (though we can't predict exactly how)
        assert not np.allclose(energies, initial_energy)

class TestUtils:
    """Test cases for utility functions."""
    
    def test_create_domain_wall(self):
        """Test creating a domain wall."""
        size = 4
        
        # Test vertical domain wall
        lattice = create_domain_wall(size, 'vertical')
        expected = np.ones((size, size))
        expected[:, size//2:] = -1
        np.testing.assert_array_equal(lattice, expected)
        
        # Test horizontal domain wall
        lattice = create_domain_wall(size, 'horizontal')
        expected = np.ones((size, size))
        expected[size//2:, :] = -1
        np.testing.assert_array_equal(lattice, expected)
    
    def test_create_checkerboard(self):
        """Test creating a checkerboard pattern."""
        size = 4
        expected = np.array([
            [1, -1, 1, -1],
            [-1, 1, -1, 1],
            [1, -1, 1, -1],
            [-1, 1, -1, 1]
        ])
        lattice = create_checkerboard(size)
        np.testing.assert_array_equal(lattice, expected)
        
        # Test with odd size (should raise error)
        with pytest.raises(ValueError):
            create_checkerboard(3)

# Run tests with: python -m pytest tests/test_ising_model.py -v
