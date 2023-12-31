use super::Vert;
use super::{face::RawFace, vertex::RawVertex};

use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct Loader {
    color: [f32; 3],
    verts: Vec<RawVertex>,
    norms: Vec<RawVertex>,
    text: Vec<RawVertex>,
    faces: Vec<RawFace>,
    invert_winding_order: bool,
}

impl Loader {
    pub fn new(file_name: &str, custom_color: [f32; 3], invert_winding_order: bool) -> Loader {
        let color = custom_color;
        let input = File::open(file_name).unwrap();
        let buffered = BufReader::new(input);

        let mut verts: Vec<RawVertex> = Vec::new();
        let mut norms: Vec<RawVertex> = Vec::new();
        let mut text: Vec<RawVertex> = Vec::new();
        let mut faces: Vec<RawFace> = Vec::new();

        for raw_line in buffered.lines() {
            let line = raw_line.unwrap();
            if line.len() > 2 {
                match line.split_at(2) {
                    ("v ", x) => {
                        verts.push(RawVertex::new(x));
                    }
                    ("vn", x) => {
                        norms.push(RawVertex::new(x));
                    }
                    ("vt", x) => {
                        text.push(RawVertex::new(x));
                    }
                    ("f ", x) => {
                        faces.push(RawFace::new(x, invert_winding_order));
                    }
                    (_, _) => {}
                };
            }
        }

        Loader {
            color,
            verts,
            norms,
            text,
            faces,
            invert_winding_order,
        }
    }

    pub fn as_normal_vertices(&self) -> Vec<Vert> {
        let mut ret: Vec<Vert> = Vec::new();
        for face in &self.faces {
            let verts = face.verts;
            let normals = face.norms.unwrap();
            ret.push(Vert {
                position: self.verts.get(verts[0]).unwrap().vals,
                normal: self.norms.get(normals[0]).unwrap().vals,
                colour: self.color,
            });
            ret.push(Vert {
                position: self.verts.get(verts[1]).unwrap().vals,
                normal: self.norms.get(normals[1]).unwrap().vals,
                colour: self.color,
            });
            ret.push(Vert {
                position: self.verts.get(verts[2]).unwrap().vals,
                normal: self.norms.get(normals[2]).unwrap().vals,
                colour: self.color,
            });
        }
        ret
    }
}
