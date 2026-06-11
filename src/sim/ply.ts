// Minimal PLY mesh parser (binary little-endian + ASCII), vertex xyz + triangulated faces.
//
// Frames: the Spaitial API's reconstructed mesh needs the (x,y,z)->(x,-y,-z)
// rotation (180° about X) into PlayCanvas display space (`transform: true`,
// the default) — verified by voxel-overlap against the splat (74-79% vs <38%
// for other candidate transforms). Splat .ply files converted with
// @playcanvas/splat-transform are already in display space — read those with
// `transform: false`.

export interface ParsedMesh {
  positions: Float32Array; // transformed, xyz triplets
  indices: Uint32Array;    // triangulated, winding fixed for the mirror transform
  aabb: { min: [number, number, number]; max: [number, number, number] };
}

type PlyType =
  | 'char' | 'uchar' | 'short' | 'ushort' | 'int' | 'uint'
  | 'float' | 'double' | 'int8' | 'uint8' | 'int16' | 'uint16'
  | 'int32' | 'uint32' | 'float32' | 'float64';

const TYPE_SIZE: Record<string, number> = {
  char: 1, int8: 1, uchar: 1, uint8: 1,
  short: 2, int16: 2, ushort: 2, uint16: 2,
  int: 4, int32: 4, uint: 4, uint32: 4,
  float: 4, float32: 4, double: 8, float64: 8,
};

interface Prop { name: string; type: PlyType; isList: boolean; countType?: PlyType }
interface Element { name: string; count: number; props: Prop[] }

function readScalar(view: DataView, offset: number, type: PlyType): number {
  switch (TYPE_SIZE[type]) {
    case 1: return type[0] === 'u' ? view.getUint8(offset) : view.getInt8(offset);
    case 2: return type[0] === 'u' ? view.getUint16(offset, true) : view.getInt16(offset, true);
    case 8: return view.getFloat64(offset, true);
    default:
      if (type === 'float' || type === 'float32') return view.getFloat32(offset, true);
      return type[0] === 'u' ? view.getUint32(offset, true) : view.getInt32(offset, true);
  }
}

export function parsePly(buffer: ArrayBuffer, opts: { transform?: boolean } = {}): ParsedMesh {
  const transform = opts.transform !== false;
  const bytes = new Uint8Array(buffer);
  // Find end_header
  const headerEnd = findHeaderEnd(bytes);
  const headerText = new TextDecoder('ascii').decode(bytes.subarray(0, headerEnd));
  const lines = headerText.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines[0] !== 'ply') throw new Error('Not a PLY file');

  let format = 'ascii' as 'ascii' | 'binary_little_endian' | 'binary_big_endian';
  const elements: Element[] = [];
  for (const line of lines) {
    const parts = line.split(/\s+/);
    if (parts[0] === 'format') format = parts[1] as typeof format;
    else if (parts[0] === 'element') elements.push({ name: parts[1], count: parseInt(parts[2], 10), props: [] });
    else if (parts[0] === 'property') {
      const el = elements[elements.length - 1];
      if (parts[1] === 'list') el.props.push({ name: parts[4], type: parts[3] as PlyType, isList: true, countType: parts[2] as PlyType });
      else el.props.push({ name: parts[2], type: parts[1] as PlyType, isList: false });
    }
  }
  if (format === 'binary_big_endian') throw new Error('big-endian PLY not supported');

  let positionsRaw: Float32Array | null = null;
  const faceIndices: number[] = [];

  if (format === 'ascii') {
    const body = new TextDecoder('ascii').decode(bytes.subarray(headerEnd));
    const tokens = body.split(/\s+/).filter(Boolean);
    let t = 0;
    for (const el of elements) {
      if (el.name === 'vertex') {
        positionsRaw = new Float32Array(el.count * 3);
        const xi = el.props.findIndex((p) => p.name === 'x');
        for (let i = 0; i < el.count; i++) {
          for (let p = 0; p < el.props.length; p++) {
            const v = parseFloat(tokens[t++]);
            if (p === xi) positionsRaw[i * 3] = v;
            else if (p === xi + 1) positionsRaw[i * 3 + 1] = v;
            else if (p === xi + 2) positionsRaw[i * 3 + 2] = v;
          }
        }
      } else {
        for (let i = 0; i < el.count; i++) {
          for (const prop of el.props) {
            if (prop.isList) {
              const n = parseInt(tokens[t++], 10);
              const face: number[] = [];
              for (let k = 0; k < n; k++) face.push(parseInt(tokens[t++], 10));
              if (el.name === 'face') pushFan(faceIndices, face);
            } else t++;
          }
        }
      }
    }
  } else {
    const view = new DataView(buffer);
    let offset = headerEnd;
    for (const el of elements) {
      if (el.name === 'vertex' && el.props.every((p) => !p.isList)) {
        const stride = el.props.reduce((s, p) => s + TYPE_SIZE[p.type], 0);
        let xOff = -1, yOff = -1, zOff = -1; let xT: PlyType = 'float', yT: PlyType = 'float', zT: PlyType = 'float';
        let acc = 0;
        for (const p of el.props) {
          if (p.name === 'x') { xOff = acc; xT = p.type; }
          if (p.name === 'y') { yOff = acc; yT = p.type; }
          if (p.name === 'z') { zOff = acc; zT = p.type; }
          acc += TYPE_SIZE[p.type];
        }
        positionsRaw = new Float32Array(el.count * 3);
        for (let i = 0; i < el.count; i++) {
          const base = offset + i * stride;
          positionsRaw[i * 3] = readScalar(view, base + xOff, xT);
          positionsRaw[i * 3 + 1] = readScalar(view, base + yOff, yT);
          positionsRaw[i * 3 + 2] = readScalar(view, base + zOff, zT);
        }
        offset += el.count * stride;
      } else {
        for (let i = 0; i < el.count; i++) {
          const face: number[] = [];
          for (const prop of el.props) {
            if (prop.isList) {
              const n = readScalar(view, offset, prop.countType!);
              offset += TYPE_SIZE[prop.countType!];
              for (let k = 0; k < n; k++) {
                face.push(readScalar(view, offset, prop.type));
                offset += TYPE_SIZE[prop.type];
              }
            } else offset += TYPE_SIZE[prop.type];
          }
          if (el.name === 'face' && face.length >= 3) pushFan(faceIndices, face);
        }
      }
    }
  }

  if (!positionsRaw) throw new Error('PLY has no vertex element');

  const positions = positionsRaw;
  if (transform) {
    for (let i = 0; i < positions.length; i += 3) {
      positions[i + 1] = -positions[i + 1];
      positions[i + 2] = -positions[i + 2];
    }
  }
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    if (x < min[0]) min[0] = x; if (y < min[1]) min[1] = y; if (z < min[2]) min[2] = z;
    if (x > max[0]) max[0] = x; if (y > max[1]) max[1] = y; if (z > max[2]) max[2] = z;
  }
  return { positions, indices: new Uint32Array(faceIndices), aabb: { min, max } };
}

function pushFan(out: number[], face: number[]) {
  for (let i = 1; i < face.length - 1; i++) out.push(face[0], face[i], face[i + 1]);
}

function findHeaderEnd(bytes: Uint8Array): number {
  const needle = 'end_header';
  const limit = Math.min(bytes.length, 65536);
  for (let i = 0; i < limit - needle.length; i++) {
    let ok = true;
    for (let k = 0; k < needle.length; k++) if (bytes[i + k] !== needle.charCodeAt(k)) { ok = false; break; }
    if (ok) {
      let j = i + needle.length;
      while (j < limit && bytes[j] !== 10) j++;
      return j + 1;
    }
  }
  throw new Error('PLY end_header not found');
}
