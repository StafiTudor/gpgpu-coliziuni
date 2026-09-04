#include "physics_object.h"
#include <glm/gtc/matrix_transform.hpp>

namespace physics {
glm::mat4 PhysicsObject::GetModelMatrix() const {
  glm::mat4 modelMatrix = glm::mat4(1.0f);
  modelMatrix = glm::translate(modelMatrix, position);
  modelMatrix = glm::scale(modelMatrix, scale);
  return modelMatrix;
}

void PhysicsObject::UpdateBoundingVolume() {
    boundingVolume.center = position;
}

void PhysicsObject::ApplyForce(const glm::vec3 &force) {
}

void PhysicsObject::ApplyImpulse(const glm::vec3 &impulse) {
}

void PhysicsObject::Integrate(float deltaTime) {
  // TODO
    if (!isStatic) {
        // mutare cub
        position += velocity * deltaTime;

        // dupa schimbarea pozitiei, actualizez si cutia lui de coliziune ca sa nu ramana inurma
        UpdateBoundingVolume();
    }
}
} // namespace physics
