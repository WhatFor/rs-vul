// use nalgebra_glm::vec4;

// use crate::{model::Model, vertex::Vert};
// use std::{fs::File, io::BufReader};

// pub fn load_model(filepath: &str) -> Model {
//     let file = File::open(filepath).expect("Could not load file");
//     let mut file_reader = BufReader::new(file);

//     let mut verticies: Vec<Vert>;

//     let (models, _) = tobj::load_obj_buf(
//         &mut file_reader,
//         &tobj::LoadOptions {
//             triangulate: true,
//             single_index: true,
//             ..Default::default()
//         },
//         |_| Ok(Default::default()),
//     )
//     .expect("Could not read model");

//     for model in &models {
//         for index in &model.mesh.indices {
//             let pos_offset = (3 * index) as usize;

//             let colour = match model.mesh.vertex_color.len() {
//                 3 => [
//                     model.mesh.vertex_color[pos_offset],
//                     model.mesh.vertex_color[pos_offset + 1],
//                     model.mesh.vertex_color[pos_offset + 2],
//                 ],
//                 _ => [0.5, 0.5, 0.5],
//             };

//             let vert = Vert {
//                 position: [
//                     model.mesh.positions[pos_offset],
//                     model.mesh.positions[pos_offset + 1],
//                     model.mesh.positions[pos_offset + 2],
//                 ],
//                 // todo: is this right? normals look fucked up atm
//                 normal: [
//                     model.mesh.normals[pos_offset],
//                     model.mesh.normals[pos_offset + 1],
//                     model.mesh.normals[pos_offset + 2],
//                 ],
//                 colour: colour,
//             };

//             verticies.push(vert);
//         }
//     }

//     Model { data: verticies }
// }
