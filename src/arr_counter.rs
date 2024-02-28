use num::BigInt;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::cmp::{max, min};
use std::fmt;
use std::io::{stdout, Write};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct VertexIndex(usize);
impl Deref for VertexIndex {
    type Target = usize;
    fn deref(&self) -> &usize {
        &self.0
    }
}
impl DerefMut for VertexIndex {
    fn deref_mut(&mut self) -> &mut usize {
        &mut self.0
    }
}
impl fmt::Debug for VertexIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct EdgeIndex(usize);
impl Deref for EdgeIndex {
    type Target = usize;
    fn deref(&self) -> &usize {
        &self.0
    }
}
impl DerefMut for EdgeIndex {
    fn deref_mut(&mut self) -> &mut usize {
        &mut self.0
    }
}
impl fmt::Debug for EdgeIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ChordId(usize);
impl Deref for ChordId {
    type Target = usize;
    fn deref(&self) -> &usize {
        &self.0
    }
}
impl DerefMut for ChordId {
    fn deref_mut(&mut self) -> &mut usize {
        &mut self.0
    }
}
impl fmt::Debug for ChordId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug)]
struct Vertex {
    id: VertexIndex,
    incident_edge: EdgeIndex,
}

impl Vertex {
    fn new(id: VertexIndex, incident_edge: EdgeIndex) -> Vertex {
        Vertex { id, incident_edge }
    }
}

#[derive(Clone, Debug)]
struct Edge {
    id: EdgeIndex,
    origin: VertexIndex,
    twin: EdgeIndex,
    prev: EdgeIndex,
    next: EdgeIndex,
    lower: bool,
    chord_id: ChordId,
    border: bool,
}

impl Edge {
    fn new(
        id: EdgeIndex,
        origin: VertexIndex,
        twin: EdgeIndex,
        prev: EdgeIndex,
        next: EdgeIndex,
        lower: bool,
        chord_id: ChordId,
    ) -> Edge {
        Edge {
            id,
            origin,
            twin,
            prev,
            next,
            lower,
            chord_id,
            border: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash)]
struct ArrChord {
    start: EdgeIndex,
    end: EdgeIndex,
    id: ChordId,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd)]
struct Matrix<T: Copy + Default> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}
impl<T: Copy + Default> Matrix<T> {
    fn new(rows: usize, cols: usize) -> Matrix<T> {
        assert!(rows > 0 && cols > 0);
        Matrix::from_iter(rows, cols, (0..).map(|_| T::default()))
    }

    fn from_iter(rows: usize, cols: usize, data: impl IntoIterator<Item = T>) -> Matrix<T> {
        assert!(rows > 0 && cols > 0);

        Matrix {
            rows,
            cols,
            data: {
                let data: Vec<_> = data.into_iter().take(rows * cols).collect();
                assert_eq!(data.len(), rows * cols);
                data
            },
        }
    }

    fn get(&self, row: usize, col: usize) -> T {
        self.data[col + row * self.cols]
    }

    fn set(&mut self, row: usize, col: usize, val: T) {
        self.data[col + row * self.cols] = val;
    }
}

impl<T: Copy + Default> Deref for Matrix<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[derive(Clone, Debug)]
struct Arrangement {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    cycle_edges: Vec<EdgeIndex>,
    valid_crossings: Matrix<bool>,
    inserted_chord: Vec<bool>,
}

impl Arrangement {
    fn new(n: usize, valid_crossings: Matrix<bool>) -> Arrangement {
        let mut arr = Arrangement {
            vertices: Vec::new(),
            edges: Vec::new(),
            cycle_edges: Vec::new(),
            valid_crossings,
            inserted_chord: vec![false; n],
        };

        let mut cycle_vertices: Vec<VertexIndex> = Vec::new();
        for _ in 0..n {
            cycle_vertices.push(arr.add_vertex());
        }

        let mut cycle_edges: Vec<EdgeIndex> = Vec::new();
        for i in 0..n {
            let e = arr.add_edge(
                cycle_vertices[i],
                cycle_vertices[(i + 1) % n],
                true,
                ChordId(0),
            );
            arr.set_border(e, true);
            arr.set_border(arr.twin(e), true);
            cycle_edges.push(e);
            arr.set_incident_edge(cycle_vertices[i], e);
        }

        for i in 0..n {
            arr.set_next(cycle_edges[i], cycle_edges[(i + 1) % n]);
            arr.set_prev(cycle_edges[(i + 1) % n], cycle_edges[i]);

            arr.set_prev(arr.twin(cycle_edges[i]), arr.twin(cycle_edges[(i + 1) % n]));
            arr.set_next(arr.twin(cycle_edges[(i + 1) % n]), arr.twin(cycle_edges[i]));
        }

        arr.cycle_edges = cycle_edges;

        arr
    }

    fn add_vertex(&mut self) -> VertexIndex {
        let v = Vertex::new(VertexIndex(self.vertices.len()), EdgeIndex(0));
        let v_id = v.id;
        self.vertices.push(v);
        v_id
    }

    fn add_edge(
        &mut self,
        origin: VertexIndex,
        destination: VertexIndex,
        lower: bool,
        chord_id: ChordId,
    ) -> EdgeIndex {
        let edge: Edge = Edge::new(
            EdgeIndex(self.edges.len()),
            origin,
            EdgeIndex(self.edges.len() + 1),
            EdgeIndex(0),
            EdgeIndex(0),
            lower,
            chord_id,
        );
        let edge_id = edge.id;
        self.edges.push(edge);
        let twin: Edge = Edge::new(
            EdgeIndex(self.edges.len()),
            destination,
            edge_id,
            EdgeIndex(0),
            EdgeIndex(0),
            !lower,
            chord_id,
        );
        self.edges.push(twin);
        edge_id
    }

    fn incident_edge(&self, v: VertexIndex) -> EdgeIndex {
        self.vertices[*v].incident_edge
    }
    fn set_incident_edge(&mut self, v: VertexIndex, e: EdgeIndex) {
        self.vertices[*v].incident_edge = e;
    }

    fn origin(&self, e: EdgeIndex) -> VertexIndex {
        self.edges[*e].origin
    }
    fn twin(&self, e: EdgeIndex) -> EdgeIndex {
        self.edges[*e].twin
    }
    fn prev(&self, e: EdgeIndex) -> EdgeIndex {
        self.edges[*e].prev
    }
    fn next(&self, e: EdgeIndex) -> EdgeIndex {
        self.edges[*e].next
    }
    fn lower(&self, e: EdgeIndex) -> bool {
        self.edges[*e].lower
    }
    fn chord_id(&self, e: EdgeIndex) -> ChordId {
        self.edges[*e].chord_id
    }
    fn border(&self, e: EdgeIndex) -> bool {
        self.edges[*e].border
    }

    fn set_origin(&mut self, e: EdgeIndex, v: VertexIndex) {
        self.edges[*e].origin = v;
    }
    //fn set_twin(&mut self, e: EdgeIndex, t: EdgeIndex) {
    //    self.edges[*e].twin = t;
    //}
    fn set_prev(&mut self, e: EdgeIndex, p: EdgeIndex) {
        self.edges[*e].prev = p;
    }
    fn set_next(&mut self, e: EdgeIndex, n: EdgeIndex) {
        self.edges[*e].next = n;
    }
    fn set_lower(&mut self, e: EdgeIndex, l: bool) {
        self.edges[*e].lower = l;
    }
    //fn set_chord_id(&mut self, e: EdgeIndex, c: ChordId) {
    //    self.edges[*e].chord_id = c;
    //}
    fn set_border(&mut self, e: EdgeIndex, b: bool) {
        self.edges[*e].border = b;
    }

    fn insert_vertex_on_edge(&mut self, e: EdgeIndex) -> VertexIndex {
        //let v: usize = self.edges[edge.twin].origin;
        let v = self.origin(self.twin(e));

        // Create the new vertex
        let new_vertex = self.add_vertex();

        // Create new edge
        let new_edge = self.add_edge(new_vertex, v, self.lower(e), self.chord_id(e));

        self.set_lower(self.twin(new_edge), self.lower(self.twin(new_edge)));

        // Update the next and previous pointers for the edges around the new vertex
        self.set_next(new_edge, self.next(e));
        self.set_prev(self.next(e), new_edge);
        self.set_prev(self.twin(new_edge), self.prev(self.twin(e)));
        self.set_next(self.prev(self.twin(e)), self.twin(new_edge));

        self.set_prev(new_edge, e);
        self.set_next(e, new_edge);
        self.set_next(self.twin(new_edge), self.twin(e));
        self.set_prev(self.twin(e), self.twin(new_edge));

        // Update the incident edges for the vertices around the new edge
        if self.incident_edge(v) == self.twin(e) {
            self.set_incident_edge(v, self.twin(new_edge));
        }

        self.set_incident_edge(new_vertex, new_edge);

        // Update the endpoint of the existing edge
        self.set_origin(self.twin(e), new_vertex);

        new_vertex
    }

    fn insert_edge(&mut self, edge1: EdgeIndex, edge2: EdgeIndex, chord_id: ChordId) -> EdgeIndex {
        // Create the new edge
        let new_edge = self.add_edge(self.origin(edge1), self.origin(edge2), false, chord_id);

        // Update the next and previous pointers of the edges around the origin and destination vertices
        self.set_next(new_edge, edge2);
        self.set_prev(new_edge, self.prev(edge1));
        self.set_next(self.twin(new_edge), edge1);
        self.set_prev(self.twin(new_edge), self.prev(edge2));

        self.set_next(self.prev(edge2), self.twin(new_edge));
        self.set_prev(edge2, new_edge);

        self.set_next(self.prev(edge1), new_edge);
        self.set_prev(edge1, self.twin(new_edge));

        new_edge
    }

    fn fill_paths(
        &self,
        arr_chord: &ArrChord,
        current_path: &mut Vec<EdgeIndex>,
        all_paths: &mut Vec<Vec<EdgeIndex>>,
    ) {
        let (mut start, end, chord_id) = (arr_chord.start, arr_chord.end, arr_chord.id);
        if !current_path.is_empty() {
            start = self.twin(current_path[current_path.len() - 1]);
        }
        let mut e = self.next(start);
        let mut exit = false;

        let mut valid_edges: Vec<EdgeIndex> = Vec::new();
        while e != start {
            if e == end {
                exit = true;
                break;
            } else if (self.lower(e) == (self.chord_id(e) < chord_id))
                && !self.border(e)
                && self.valid_crossings.get(*self.chord_id(e), *chord_id)
            {
                valid_edges.push(e);
            }

            e = self.next(e);
        }

        if exit {
            all_paths.push(current_path.clone());
        } else {
            //valid_edges.shuffle(&mut thread_rng());
            for e_id in valid_edges {
                current_path.push(e_id);
                self.fill_paths(arr_chord, current_path, all_paths);
                current_path.pop();
            }
        }
    }

    fn fill_random_path(&self, arr_chord: &ArrChord, current_path: &mut Vec<EdgeIndex>) {
        let (mut start, end, chord_id) = (arr_chord.start, arr_chord.end, arr_chord.id);
        if !current_path.is_empty() {
            start = self.twin(current_path[current_path.len() - 1]);
        }
        let mut e = self.next(start);
        let mut exit = false;

        let mut valid_edges: Vec<EdgeIndex> = Vec::new();
        while e != start {
            if e == end {
                exit = true;
                break;
            } else if (self.lower(e) == (self.chord_id(e) < chord_id))
                && !self.border(e)
                && self.valid_crossings.get(*self.chord_id(e), *chord_id)
            {
                valid_edges.push(e);
            }

            e = self.next(e);
        }
        if !exit {
            let e_id = valid_edges[thread_rng().gen::<usize>() % valid_edges.len()];
            current_path.push(e_id);
            self.fill_random_path(arr_chord, current_path);
        }
    }

    fn count_paths(&self, arr_chord: ArrChord) -> i64 {
        let (start, end, chord_id) = (arr_chord.start, arr_chord.end, arr_chord.id);

        let mut incident_face: Vec<usize> = vec![0; self.edges.len()];
        let mut num_faces: usize = 1;
        for edge in 0..self.edges.len() {
            let edge = EdgeIndex(edge);
            if incident_face[*edge] == 0 {
                incident_face[*edge] = num_faces;
                let mut e = self.next(edge);
                while e != edge {
                    incident_face[*e] = num_faces;
                    e = self.next(e);
                }
                num_faces += 1;
            }
        }

        let mut pred_faces: Vec<Vec<usize>> = vec![Vec::with_capacity(8); num_faces];
        let mut num_succ_faces: Vec<u32> = vec![0; num_faces];

        for edge in 0..self.edges.len() {
            let edge = EdgeIndex(edge);
            if (self.lower(edge) == (self.chord_id(edge) < chord_id))
                && !self.border(edge)
                && self.valid_crossings.get(*self.chord_id(edge), *chord_id)
            {
                let f1 = incident_face[*edge];
                let f2 = incident_face[*self.twin(edge)];
                pred_faces[f2].push(f1);
                num_succ_faces[f1] += 1;
            }
        }

        let mut no_succ: Vec<usize> = Vec::with_capacity(8);
        no_succ.push(incident_face[*end]);

        let mut inverse_top_order: Vec<usize> = Vec::with_capacity(num_faces);
        while !no_succ.is_empty() {
            if let Some(f) = no_succ.pop() {
                for pred_f in &pred_faces[f] {
                    num_succ_faces[*pred_f] -= 1;
                    if num_succ_faces[*pred_f] == 0 {
                        no_succ.push(*pred_f);
                    }
                }
                inverse_top_order.push(f);
                if f == incident_face[*start] {
                    break;
                }
            }
        }

        //let big_one: BigInt = BigInt::new(Sign::Plus, vec![1]);
        //let mut num_paths: Vec<BigInt> = vec![-big_one.clone(); num_faces];
        let mut num_paths: Vec<i64> = vec![0; num_faces];
        num_paths[incident_face[*start]] = 1;

        for f in inverse_top_order.iter().rev() {
            let mut num_paths_to_f = 0;
            for f2 in &pred_faces[*f] {
                num_paths_to_f += num_paths[*f2]; //need clone if bignum
            }
            num_paths[*f] += num_paths_to_f;
        }
        num_paths[incident_face[*end]] //need clone if bignum
    }

    fn insert_path(&mut self, arr_chord: ArrChord, path: &[EdgeIndex]) {
        let (start, end, chord_id) = (arr_chord.start, arr_chord.end, arr_chord.id);
        self.inserted_chord[*chord_id] = true;
        let mut edge_pairs: Vec<(EdgeIndex, EdgeIndex)> = Vec::with_capacity(path.len() + 3);
        if !path.is_empty() {
            let mut new_vertices: Vec<VertexIndex> = Vec::with_capacity(path.len());
            for e in path {
                new_vertices.push(self.insert_vertex_on_edge(*e));
            }
            edge_pairs.push((start, self.incident_edge(new_vertices[0])));
            for i in 0..(path.len() - 1) {
                edge_pairs.push((
                    self.next(self.twin(self.incident_edge(new_vertices[i]))),
                    self.incident_edge(new_vertices[i + 1]),
                ));
            }
            edge_pairs.push((
                self.next(self.twin(self.incident_edge(new_vertices[new_vertices.len() - 1]))),
                end,
            ));
        } else {
            edge_pairs.push((start, end));
        }
        for (e1, e2) in edge_pairs {
            self.insert_edge(e1, e2, chord_id);
        }
    }

    fn delete_path(&mut self, m: usize, id: ChordId) {
        self.inserted_chord[*id] = false;
        let l = self.edges.len() - 1;

        for i in 0..(m + 1) {
            let e: EdgeIndex = EdgeIndex(l - 2 * i);
            self.set_prev(self.next(e), self.prev(self.twin(e)));
            self.set_next(self.prev(self.twin(e)), self.next(e));
            self.set_next(self.prev(e), self.next(self.twin(e)));
            self.set_prev(self.next(self.twin(e)), self.prev(e));
        }

        let l = self.vertices.len() - 1;
        for i in 0..m {
            let v: VertexIndex = VertexIndex(l - i);
            let ne = self.incident_edge(v);
            let e = self.prev(ne);
            let w = self.origin(self.twin(ne));
            self.set_next(e, self.next(ne));
            self.set_prev(self.twin(e), self.prev(self.twin(ne)));
            self.set_origin(self.twin(e), w);
            self.set_prev(self.next(ne), e);
            self.set_next(self.prev(self.twin(ne)), self.twin(e));
            if self.incident_edge(w) == self.twin(ne) {
                self.set_incident_edge(w, self.twin(e));
            }
        }
        self.edges.truncate(self.edges.len() - 4 * m - 2);
        if m > 0 {
            self.vertices.truncate(self.vertices.len() - m);
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Chord {
    start: usize,
    end: usize,
    id: ChordId,
}

fn chords_cross(chord1: &Chord, chord2: &Chord) -> bool {
    let (s1, e1) = (min(chord1.start, chord1.end), max(chord1.start, chord1.end));
    let (s2, e2) = (min(chord2.start, chord2.end), max(chord2.start, chord2.end));
    (s1 >= s2 || s2 >= e2 || e2 >= e1) && ((s1 < s2 && s2 < e1) || (s1 < e2 && e2 < e1))
}

fn compute_crossings(chords: &Vec<Chord>) -> Matrix<bool> {
    let n = chords.len();
    let mut crossings: Matrix<bool> = Matrix::new(n, n);
    for i1 in 0..n {
        for i2 in 0..n {
            if i1 == i2 {
                continue;
            }
            let (chord1, chord2) = (chords[i1], chords[i2]);
            if chords_cross(&chord1, &chord2) {
                crossings.set(*chord1.id, *chord2.id, true);
            }
        }
    }
    crossings
}

fn get_dependence_graph(chords: &Vec<Chord>, crossings: &Matrix<bool>) -> Vec<Vec<ChordId>> {
    let n = chords.len();

    let mut adj: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];

    for chord1 in chords.iter() {
        for chord2 in chords.iter() {
            if crossings.get(*chord1.id, *chord2.id) {
                adj[*chord1.id].insert(*chord2.id);
                adj[*chord2.id].insert(*chord1.id);
            }
        }
    }

    let mut dependent_adj: Vec<Vec<ChordId>> = vec![Vec::new(); n];

    let mut dp: Matrix<usize> = Matrix::new(2 * n + 1, 2 * n + 1);
    {
        let mut m: Matrix<usize> = Matrix::new(2 * n, 2 * n);
        for chord in chords.iter() {
            m.set(chord.start, chord.end, 1);
            m.set(chord.end, chord.start, 1);
        }

        for row in 1..(2 * n + 1) {
            for col in 1..(2 * n + 1) {
                let val = dp.get(row - 1, col) + dp.get(row, col - 1) - dp.get(row - 1, col - 1)
                    + m.get(row - 1, col - 1);
                dp.set(row, col, val);
            }
        }
    }

    for left_chord in chords {
        let (left_u, left_v) = (left_chord.start, left_chord.end);
        let (left_u, left_v) = (min(left_u, left_v), max(left_u, left_v));
        for right_chord in chords {
            let mut independent;

            let (right_u, right_v) = (right_chord.start, right_chord.end);
            let (right_u, right_v) = (min(right_u, right_v), max(right_u, right_v));
            if right_chord.id == left_chord.id || left_u > right_u {
                continue;
            }
            if crossings.get(*left_chord.id, *right_chord.id) {
                independent = true;
                for n_id in &adj[*left_chord.id] {
                    if crossings.get(*right_chord.id, *n_id) {
                        independent = false;
                        break;
                    }
                }
                if independent {
                    continue;
                }
            } else {
                if left_u < right_u && right_u < right_v && right_v < left_v {
                    if dp.get(right_u, left_v) + dp.get(left_u + 1, right_v + 1)
                        - dp.get(right_u, right_v + 1)
                        - dp.get(left_u + 1, left_v)
                        > 0
                    {
                        continue;
                    }
                } else {
                    if dp.get(left_u, right_u) + dp.get(0, left_v + 1)
                        - dp.get(left_u, left_v + 1)
                        - dp.get(0, right_u)
                        > 0
                    {
                        continue;
                    }
                    if dp.get(2 * n, right_u) + dp.get(right_v + 1, left_v + 1)
                        - dp.get(2 * n, left_v + 1)
                        - dp.get(right_v + 1, right_u)
                        > 0
                    {
                        continue;
                    }
                }
                independent = true;
                for c1 in &adj[*left_chord.id] {
                    if !independent {
                        break;
                    };
                    for c2 in &adj[*right_chord.id] {
                        if c1 != c2
                            && crossings.get(*c1, *c2)
                            && crossings.get(*c1, *right_chord.id)
                            && crossings.get(*c2, *left_chord.id)
                        {
                            independent = false;
                            break;
                        }
                    }
                }
            }

            if !independent {
                dependent_adj[*left_chord.id].push(right_chord.id);
                dependent_adj[*right_chord.id].push(left_chord.id);
            }
        }
    }
    dependent_adj
}

struct ArrangementSender {
    arrangement: Arrangement,
    chords: Vec<ArrChord>,
    tx: Sender<Arrangement>,
}

impl ArrangementSender {
    fn start(&mut self, maxdepth: usize, depth: usize) {
        if depth >= maxdepth {
            if self.tx.send(self.arrangement.clone()).is_err() {
                panic!();
            }
        } else {
            let mut temp_path: Vec<EdgeIndex> = Vec::new();
            let mut all_paths: Vec<Vec<EdgeIndex>> = Vec::new();
            self.arrangement
                .fill_paths(&self.chords[depth], &mut temp_path, &mut all_paths);

            all_paths.shuffle(&mut thread_rng());

            for path in &all_paths {
                self.arrangement.insert_path(self.chords[depth], path);
                self.start(maxdepth, depth + 1);
                self.arrangement
                    .delete_path(path.len(), self.chords[depth].id);
            }
        }
    }
}

struct ArrangementIterator {
    rx: Receiver<Arrangement>,
}

impl ArrangementIterator {
    fn new(arrangement: &Arrangement, chords: &[ArrChord], maxdepth: usize) -> Self {
        let (tx, rx): (Sender<Arrangement>, Receiver<Arrangement>) = mpsc::channel();
        let arrangement = arrangement.clone();
        let chords = chords.to_vec();
        thread::spawn(move || {
            ArrangementSender {
                arrangement,
                chords,
                tx,
            }
            .start(maxdepth, 0)
        });
        ArrangementIterator { rx }
    }
}

impl Iterator for ArrangementIterator {
    type Item = Arrangement;
    fn next(&mut self) -> Option<Self::Item> {
        match self.rx.recv() {
            Ok(arr) => Some(arr),
            Err(_) => None,
        }
    }
}

fn count_in_random_arrangement<'a>(
    chord: ArrChord,
    other_chords: &'a Vec<ArrChord>,
    arrangement: &'a mut Arrangement,
) -> i64 {
    let mut all_paths_len: Vec<(usize, ChordId)> = Vec::new();
    for chord_b in other_chords {
        let mut path: Vec<EdgeIndex> = Vec::new();
        arrangement.fill_random_path(chord_b, &mut path);
        arrangement.insert_path(*chord_b, &path);
        all_paths_len.push((path.len(), chord_b.id));
    }
    let t = arrangement.count_paths(chord);

    for &(l, id) in all_paths_len.iter().rev() {
        arrangement.delete_path(l, id);
    }
    t
}

fn estimate_line_contribution(
    chord_id: ChordId,
    chords: &Vec<ArrChord>,
    arrangement: &mut Arrangement,
) -> f64 {
    let mut other_chords: Vec<ArrChord> = Vec::new();
    for chord in chords {
        if chord.id != chord_id {
            other_chords.push(*chord);
        }
    }
    let mut t: i64 = 0;
    for _ in 0..200 {
        t += count_in_random_arrangement(chords[*chord_id], &other_chords, arrangement);
    }
    (t as f64).log2()
}
#[derive(Debug, Clone)]
struct ChordPartition {
    root: Vec<ChordId>,
    children: Option<Box<(ChordPartition, ChordPartition)>>,
}

impl ChordPartition {
    fn new(root: Vec<ChordId>, children: Option<Box<(ChordPartition, ChordPartition)>>) -> Self {
        ChordPartition { root, children }
    }
    fn delete_first_k(&mut self, k: usize) {
        self.root.drain(0..k);
    }
}

fn get_partition(
    weights: &Vec<f64>,
    dependent_adj: &Vec<Vec<ChordId>>,
    chords: &mut [ChordId],
) -> ChordPartition {
    let mut best_root = chords.to_vec();
    let mut best_left: Vec<ChordId> = Vec::new();
    let mut best_right: Vec<ChordId> = Vec::new();

    let mut best_min_weight = -1.0;
    let mut best_max_weight = -1.0;

    for i1 in 0..chords.len() {
        for i2 in (i1 + 1)..chords.len() {
            for i3 in (i2 + 1)..chords.len() {
                for i4 in (i3 + 1)..chords.len() {
            
                    let seed_left = chords[i1];
                    let seed_right = chords[i2];
                    let seed_left2 = chords[i3];
                    let seed_right2 = chords[i4];
                    if  dependent_adj[*seed_left].contains(&seed_right)  ||
                        dependent_adj[*seed_left].contains(&seed_right2) ||
                        dependent_adj[*seed_left2].contains(&seed_right) ||
                        dependent_adj[*seed_left2].contains(&seed_right2)
                    {
                        continue;
                    }

                    for _ in 0..50 {
                        let mut root: Vec<ChordId> = Vec::new();
                        let mut left = vec![seed_left, seed_left2];
                        let mut right = vec![seed_right, seed_right2];
                        let mut left_weight = weights[*seed_left] + weights[*seed_left2];
                        let mut right_weight = weights[*seed_right] + weights[*seed_right2];

                        chords.shuffle(&mut thread_rng());
                        for i in chords.iter() {
                            let i = *i;
                            if i == seed_left || i == seed_right || i == seed_left2 || i == seed_right2 {
                                continue;
                            }

                            let insertable_left = !right.iter().any(|&x| dependent_adj[*x].contains(&i));
                            let insertable_right = !left.iter().any(|&x| dependent_adj[*x].contains(&i));

                            if !insertable_left && !insertable_right {
                                root.push(i);
                            } else if !insertable_right || (insertable_left && left_weight <= right_weight)
                            {
                                left.push(i);
                                left_weight += weights[*i];
                            } else {
                                right.push(i);
                                right_weight += weights[*i];
                            }
                        }

                        let (min_weight, max_weight) = if left_weight < right_weight {
                            (left_weight, right_weight)
                        } else {
                            (right_weight, left_weight)
                        };
                        if min_weight > best_min_weight
                            || (min_weight == best_min_weight && max_weight > best_max_weight)
                        {
                            best_min_weight = min_weight;
                            best_max_weight = max_weight;
                            best_left = left.clone();
                            best_right = right.clone();
                            best_root = root.clone();
                        }
                    }
                }
            }
        }
    }

    for i1 in 0..chords.len() {
        for i2 in (i1 + 1)..chords.len() {
           
            
                    let seed_left = chords[i1];
                    let seed_right = chords[i2];
                    if  dependent_adj[*seed_left].contains(&seed_right)
                    {
                        continue;
                    }

                    for _ in 0..((chords.len()+10)*(chords.len()+10)) {
                        let mut root: Vec<ChordId> = Vec::new();
                        let mut left = vec![seed_left];
                        let mut right = vec![seed_right];
                        let mut left_weight = weights[*seed_left];
                        let mut right_weight = weights[*seed_right];

                        chords.shuffle(&mut thread_rng());
                        for i in chords.iter() {
                            let i = *i;
                            if i == seed_left || i == seed_right {
                                continue;
                            }

                            let insertable_left = !right.iter().any(|&x| dependent_adj[*x].contains(&i));
                            let insertable_right = !left.iter().any(|&x| dependent_adj[*x].contains(&i));

                            if !insertable_left && !insertable_right {
                                root.push(i);
                            } else if !insertable_right || (insertable_left && left_weight <= right_weight)
                            {
                                left.push(i);
                                left_weight += weights[*i];
                            } else {
                                right.push(i);
                                right_weight += weights[*i];
                            }
                        }

                        let (min_weight, max_weight) = if left_weight < right_weight {
                            (left_weight, right_weight)
                        } else {
                            (right_weight, left_weight)
                        };
                        if min_weight > best_min_weight
                            || (min_weight == best_min_weight && max_weight > best_max_weight)
                        {
                            best_min_weight = min_weight;
                            best_max_weight = max_weight;
                            best_left = left.clone();
                            best_right = right.clone();
                            best_root = root.clone();
                        }

            }
        }
    }

    best_left.sort_by(|&a, &b| weights[*a].total_cmp(&weights[*b]));
    best_right.sort_by(|&a, &b| weights[*a].total_cmp(&weights[*b]));
    best_root.sort_by(|&a, &b| weights[*a].total_cmp(&weights[*b]));

    //let left_partition: ChordPartition;
    //let right_partition: ChordPartition;

    if best_left.is_empty() || best_right.is_empty() {
        chords.sort_by(|&a, &b| weights[*a].total_cmp(&weights[*b]));
        return ChordPartition::new(chords.to_vec(), None);
    }

    let left_partition = if best_left.len() > 1 && best_left.len() < chords.len() {
        get_partition(weights, dependent_adj, best_left.as_mut_slice())
    } else {
        ChordPartition::new(best_left, None)
    };
    let right_partition = if best_right.len() > 1 && best_right.len() < chords.len() {
        get_partition(weights, dependent_adj, best_right.as_mut_slice())
    } else {
        ChordPartition::new(best_right, None)
    };

    ChordPartition::new(best_root, Some(Box::new((left_partition, right_partition))))
}

/* ---- */

fn simplify_chords(chords: Vec<Chord>) -> Vec<Chord> {
    let crossings = compute_crossings(&chords);
    let mut simplified_chords: Vec<Chord> = Vec::new();
    for chord in chords.iter() {
        let (u, v, chord_id) = (chord.start, chord.end, chord.id);
        let mut crossed: Vec<ChordId> = Vec::new();
        for chord_b in chords.iter() {
            if chord_b.id != chord_id && crossings.get(*chord_id, *chord_b.id) {
                crossed.push(chord_b.id);
            }
        }
        let mut to_include = false;
        for chord_a_id in crossed.iter() {
            let chord_a_id = *chord_a_id;
            if to_include {
                break;
            }
            for chord_b_id in crossed.iter() {
                let chord_b_id = *chord_b_id;
                if chord_a_id != chord_b_id && crossings.get(*chord_a_id, *chord_b_id) {
                    to_include = true;
                    break;
                }
            }
        }
        if to_include {
            simplified_chords.push(Chord {
                start: u,
                end: v,
                id: ChordId(simplified_chords.len()),
            });
        }
    }

    let mut relevant_endpoints: Vec<usize> = Vec::new();
    for chord in simplified_chords.iter() {
        relevant_endpoints.push(chord.start);
        relevant_endpoints.push(chord.end);
    }

    simplified_chords.sort();
    relevant_endpoints.sort();

    let mut endpoint_to_index: Vec<usize> = vec![0; 2 * chords.len()];
    for (i, &end_id) in relevant_endpoints.iter().enumerate() {
        endpoint_to_index[end_id] = i;
    }

    for (i, chord) in simplified_chords.iter_mut().enumerate() {
        chord.start = endpoint_to_index[chord.start];
        chord.end = endpoint_to_index[chord.end];
        chord.id = ChordId(i);
    }

    simplified_chords
}

fn count_helper(
    chords: &Vec<ArrChord>,
    partition: &ChordPartition,
    arrangement: &mut Arrangement,
    depth: usize,
    timeout: Option<Instant>,
) -> BigInt {
    let mut total: BigInt = BigInt::from(0);

    match timeout{
        Some(timeout) =>   if Instant::now() > timeout {
                                        return BigInt::from(0);
                                    },
        None => {},
    }
    

    match &partition.children {
        Some(children) => {
            let root = &partition.root;
            let left = &children.0;
            let right = &children.1;
            if depth >= root.len() {
                let t1 = count_helper(chords, left, arrangement, 0, timeout);
                let t2 = count_helper(chords, right, arrangement, 0, timeout);
                return t1 * t2;
            } else {
                let mut all_paths: Vec<Vec<EdgeIndex>> = Vec::new();
                let mut current_path: Vec<EdgeIndex> = Vec::new();
                arrangement.fill_paths(&chords[*root[depth]], &mut current_path, &mut all_paths);

                for path in all_paths {
                    let chord = chords[*root[depth]];
                    arrangement.insert_path(chord, &path);
                    total += count_helper(chords, partition, arrangement, depth + 1, timeout);
                    arrangement.delete_path(path.len(), chord.id);

                    match timeout{
                        Some(timeout) =>   if Instant::now() > timeout {
                                                        return BigInt::from(0);
                                                    },
                        None => {},
                    };
                }
            }
        }
        None => {
            let root = &partition.root;

            match depth + 1 {
                x if x < root.len() => {
                    let mut all_paths: Vec<Vec<EdgeIndex>> = Vec::new();
                    let mut current_path: Vec<EdgeIndex> = Vec::new();
                    arrangement.fill_paths(
                        &chords[*root[depth]],
                        &mut current_path,
                        &mut all_paths,
                    );

                    for path in all_paths {
                        let chord = chords[*root[depth]];
                        arrangement.insert_path(chord, &path);
                        total += count_helper(chords, partition, arrangement, depth + 1, timeout);
                        arrangement.delete_path(path.len(), chord.id);
                        match timeout{
                            Some(timeout) =>   if Instant::now() > timeout {
                                                            return BigInt::from(0);
                                                        },
                            None => {},
                        }
                    }
                }
                x if x > root.len() => {
                    total = BigInt::from(1);
                }
                _ => {
                    total = BigInt::from(arrangement.count_paths(chords[*root[root.len() - 1]]));
                }
            }
        }
    }
    match timeout{
        Some(timeout) =>   if Instant::now() > timeout {
                                        return BigInt::from(0);
                                    }else {
                                        return total;
                                    },
        None => return total,
    }
}

fn time_string(mut seconds: i64) -> String {
    let h = seconds / 3600;
    seconds -= 3600 * h;
    let m = seconds / 60;
    seconds -= 60 * m;
    format!("{}h {}m {}s", h, m, seconds)
}

pub fn count(chords: Vec<(usize, usize)>, time_ceil: Option<i64>) -> Option<BigInt> {

    let time_ceil_value : i64;
    match time_ceil{
        Some(time_ceil) => time_ceil_value = time_ceil,
        None => time_ceil_value = 0,
    }

    let chords: Vec<Chord> = chords
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| Chord {
            start: x,
            end: y,
            id: ChordId(i),
        })
        .collect();

    let chords = simplify_chords(chords);
    let mut m = 0;
    for chord in chords.iter() {
        m = max(m, max(chord.start, chord.end));
    }
    m += 1;

    let crossings = compute_crossings(&chords);
    let mut arrangement = Arrangement::new(m, crossings.clone());

    let mut arr_chords: Vec<ArrChord> = Vec::new();
    for chord in chords.iter() {
        arr_chords.push(ArrChord {
            start: arrangement.cycle_edges[chord.start],
            end: arrangement.cycle_edges[chord.end],
            id: chord.id,
        });
    }

    println!("Computing dependency and weights...");
    let adj = get_dependence_graph(&chords, &crossings);

    let weights: Vec<_> = (0..chords.len())
        .map(|i| estimate_line_contribution(ChordId(i), &arr_chords, &mut arrangement))
        .collect();

    println!("Computing partition...");

    let mut chord_ids: Vec<_> = chords.iter().map(|c| c.id).collect();
    let mut partition = get_partition(&weights, &adj, &mut chord_ids);

    let base_ids = partition.root.clone();
    let base_chords: Vec<_> = base_ids.iter().map(|i| arr_chords[**i]).collect();

    let mut depth = max(1, min(base_chords.len(), 6));

    let mut tot_at_depth: u64 = ArrangementIterator::new(&arrangement, &base_chords, depth)
        .map(|_| 1)
        .sum();

    while tot_at_depth < 1000_u64 && depth < base_chords.len() {
        depth += 1;
        tot_at_depth = ArrangementIterator::new(&arrangement, &base_chords, depth)
            .map(|_| 1)
            .sum();
    }

    partition.delete_first_k(depth);

    println!("\nStarting count with {} independent jobs:", tot_at_depth);
    print!("0%, Elapsed: 0s, Remaining (est.): ???, Estimated total: ???");
    stdout().flush().expect("Can't flush output");

    let tot_at_depth = tot_at_depth;
    let steps = AtomicI64::new(0);
    let fail = AtomicBool::new(false);
    let estimated_time = AtomicI64::new(0);
    let total = Arc::new(Mutex::new(BigInt::from(0)));

    let st = Instant::now();

    ArrangementIterator::new(&arrangement, &base_chords, depth)
        .par_bridge() //
        .for_each(|mut arr| {
            if fail.load(Ordering::Relaxed)
                || (time_ceil_value > 0 && steps.load(Ordering::Relaxed) >= 7 && estimated_time.load(Ordering::Relaxed) > time_ceil_value)
            {
                fail.store(true, Ordering::Relaxed);
                return;
            }

            let timeout: Option<Instant>;
            if time_ceil_value > 0{
                if steps.load(Ordering::Relaxed) < 1{
                    timeout = Some(min(st + Duration::from_secs(time_ceil_value as u64), Instant::now() + Duration::from_secs(5 + 8 * (time_ceil_value as u64) / tot_at_depth)));
                }else{
                    timeout = Some(min(st + Duration::from_secs(time_ceil_value as u64), Instant::now() + Duration::from_secs(5 + 30 * (time_ceil_value as u64) / tot_at_depth)));
                };
            }else{
                timeout = None;
            }

            let r = count_helper(
                &arr_chords,
                &partition,
                &mut arr,
                0,
                timeout,
            );
            if r == BigInt::from(0) {
                fail.store(true, Ordering::Relaxed);
                return;
            }
            let mut total = total.lock().unwrap();
            *total += r.clone();
            steps.fetch_add(1, Ordering::Relaxed);

            let steps = steps.load(Ordering::Relaxed) as u64;
            let prog = 100.0 * (steps as f64) / (tot_at_depth as f64);
            let tot_time = (Instant::now() - st).as_secs() as f64;
            let remaining = (100.0 - prog) * (tot_time) / prog;
            estimated_time.store((tot_time + remaining) as i64, Ordering::Relaxed);
            let estimated_total =
                (BigInt::from(tot_at_depth) * (*total).clone()) / BigInt::from(steps);
            print!(
                "\r{:.2}%, Elapsed: {}, Remaining (est.): {}, Estimated total: {}            ",
                prog,
                time_string(tot_time as i64),
                time_string(remaining as i64),
                estimated_total,
            );
            stdout().flush().expect("Can't flush output");
        });

    println!("\n");
    if fail.load(Ordering::Relaxed) {
        None
    } else {
        let total = (*total.lock().unwrap()).clone();
        Some(total)
    }
}
