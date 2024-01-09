use super::obj_loader::{ColoredVertex, Loader, Vert};

use nalgebra_glm::{
    identity, inverse_transpose, rotate_normalized_axis, scale, translate, vec3, TMat4, TVec3,
};

pub struct Model {
    data: Vec<Vert>,
    translation: TMat4<f32>,
    velocity: TVec3<f32>,
    forces: Vec<TVec3<f32>>,
    uniform_scale: f32,
    rotation: TMat4<f32>,
    model: TMat4<f32>,
    normals: TMat4<f32>,
    requires_update: bool,
}

pub struct ModelBuilder {
    file_name: String,
    scale_factor: f32,
    custom_color: [f32; 3],
    invert: bool,
}

impl ModelBuilder {
    fn new(file: String) -> ModelBuilder {
        ModelBuilder {
            file_name: file,
            custom_color: [1.0, 0.35, 0.137],
            invert: true,
            scale_factor: 1.,
        }
    }

    pub fn build(self) -> Model {
        let loader = Loader::new(self.file_name.as_str(), self.custom_color, self.invert);
        Model {
            data: loader.as_normal_vertices(),
            translation: identity(),
            velocity: vec3(0.0, 0.0, 0.0),
            forces: Vec::new(),
            uniform_scale: self.scale_factor,
            rotation: identity(),
            model: identity(),
            normals: identity(),
            requires_update: false,
        }
    }

    pub fn color(mut self, new_color: [f32; 3]) -> ModelBuilder {
        self.custom_color = new_color;
        self
    }

    pub fn file(mut self, file: String) -> ModelBuilder {
        self.file_name = file;
        self
    }

    pub fn invert_winding_order(mut self, invert: bool) -> ModelBuilder {
        self.invert = invert;
        self
    }

    pub fn uniform_scale_factor(mut self, scale: f32) -> ModelBuilder {
        self.scale_factor = scale;
        self
    }
}

impl Model {
    pub fn new(file_name: &str) -> ModelBuilder {
        ModelBuilder::new(file_name.into())
    }

    pub fn data(&self) -> Vec<Vert> {
        self.data.clone()
    }

    pub fn model_matrices(&mut self) -> (TMat4<f32>, TMat4<f32>) {
        if self.requires_update {
            self.model = self.translation * self.rotation;

            self.model = scale(
                &self.model,
                &vec3(self.uniform_scale, self.uniform_scale, self.uniform_scale),
            );

            self.normals = inverse_transpose(self.model);
            self.requires_update = false;
        }
        (self.model, self.normals)
    }

    pub fn rotate(&mut self, radians: f32, v: TVec3<f32>) {
        self.rotation = rotate_normalized_axis(&self.rotation, radians, &v);
        self.requires_update = true;
    }

    pub fn translate(&mut self, v: TVec3<f32>) {
        self.translation = translate(&self.translation, &v);
        self.requires_update = true;
    }

    pub fn reset_translation(&mut self) {
        self.translation = identity();
        self.requires_update = true;
    }

    pub fn reset_velocity(&mut self) {
        self.velocity = vec3(0.0, 0.0, 0.0);
    }

    pub fn set_velocity(&mut self, velocity: TVec3<f32>) {
        self.velocity = velocity;
    }

    pub fn add_gravity(&mut self) {
        self.add_force(vec3(0.0, 9.81, 0.0));
    }

    pub fn add_force(&mut self, force: TVec3<f32>) {
        self.forces.push(force);
        self.requires_update = true;
    }

    pub fn apply_forces(&mut self, delta_time: f32) {
        let mut total_force = vec3(0.0, 0.0, 0.0);

        for force in &self.forces {
            total_force += force;
        }

        self.velocity += total_force * delta_time;

        let translation = self.velocity * delta_time;
        self.translate(translation);
    }

    pub fn scale(&mut self, scale: f32) {
        self.uniform_scale = scale;
        self.requires_update = true;
    }

    /// Return the model's rotation to 0
    pub fn zero_rotation(&mut self) {
        self.rotation = identity();
        self.requires_update = true;
    }

    pub fn color_data(&self) -> Vec<ColoredVertex> {
        let mut ret: Vec<ColoredVertex> = Vec::new();
        for v in &self.data {
            ret.push(ColoredVertex {
                position: v.position,
                color: v.colour,
            });
        }
        ret
    }
}
