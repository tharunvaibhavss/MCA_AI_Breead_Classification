'use client';

import React, { useState, useEffect } from 'react';
import { Database, Search, MapPin, Milk, ThermometerSun, ShieldCheck } from 'lucide-react';
import { fetchBreedsList, BreedInfo } from '@/lib/api';

export default function BreedsPage() {
  const [breeds, setBreeds] = useState<BreedInfo[]>([]);
  const [filter, setFilter] = useState<'all' | 'cattle' | 'buffalo'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);

  const defaultBreedDetails: Record<string, any> = {
    Dangi: {
      native_state: 'Maharashtra / Gujarat',
      origin: 'Nashik, Ahmednagar & Dang Districts',
      characteristics: 'Distinctive white coat with uneven black or red spots, hardy hooves adapted to heavy rainfall and stony terrains.',
      milk_yield: '530 kg per lactation',
      fat_percentage: '4.3% - 4.5%',
      uses: 'Hardy Draught Power',
      climate: 'Exceptional endurance in heavy monsoon and hilly timber terrain.',
    },
    Gaolao: {
      native_state: 'Maharashtra / Madhya Pradesh',
      origin: 'Wardha, Nagpur & Chhindwara Region',
      characteristics: 'White or light grey coat, convex forehead, almond-shaped eyes, agile muscular carriage.',
      milk_yield: '800 kg per lactation',
      fat_percentage: '4.2% - 4.6%',
      uses: 'Dual-Purpose (Draught & Milk)',
      climate: 'Well adapted to dry deciduous agro-climatic conditions.',
    },
    Gir: {
      native_state: 'Gujarat',
      origin: 'Gir Hills & Kathiawar Forests',
      characteristics: 'Distinctive rounded forehead, long pendulous leaf-like ears, curved horns turning upward.',
      milk_yield: '2,182 kg per lactation',
      fat_percentage: '4.5% - 5.0%',
      uses: 'Dairy (A2 Milk)',
      climate: 'Exceptional heat tolerance and tick parasite resistance.',
    },
    Ongole: {
      native_state: 'Andhra Pradesh',
      origin: 'Ongole Taluk, Prakasam District',
      characteristics: 'Glossy white coat, muscular hump, broad flat forehead, short stumpy horns.',
      milk_yield: '1,500 kg per lactation',
      fat_percentage: '4.2% - 4.8%',
      uses: 'Dual-Purpose (Draft & Dairy)',
      climate: 'Superior tropical heat endurance and heavy draft work capability.',
    },
    Sahiwal: {
      native_state: 'Punjab',
      origin: 'Sahiwal / Montgomery Region',
      characteristics: 'Reddish-brown coat, medium broad flat forehead, short thick horns, loose skin.',
      milk_yield: '2,325 kg per lactation',
      fat_percentage: '4.8% - 5.2%',
      uses: 'Dairy',
      climate: 'High resistance to extreme heat and tropical tick infestations.',
    },
    Manda: {
      native_state: 'Odisha / Andhra Pradesh',
      origin: 'Eastern Ghats / Koraput Plateau',
      characteristics: 'Ash-grey or copper-brown coat, broad face, short backwardly directed horns, sturdy draft build.',
      milk_yield: '600 - 800 kg per lactation',
      fat_percentage: '7.8% - 8.5%',
      uses: 'Dual-Purpose (Draft & High-Fat Dairy)',
      climate: 'Highly resilient to hill terrain and endemic parasitic stress.',
    },
    Surti: {
      native_state: 'Gujarat',
      origin: 'Kheda & Vadodara Region',
      characteristics: 'Compact medium frame, sickle-shaped horns, rusty brown coat with two white neck collars.',
      milk_yield: '1,650 kg per lactation',
      fat_percentage: '7.5% - 8.2%',
      uses: 'Economical Dairy',
      climate: 'Thrives in intensive rural management with minimal green forage.',
    },
    Toda: {
      native_state: 'Tamil Nadu',
      origin: 'Nilgiri Hills (High Altitude Shola Grasslands)',
      characteristics: 'Dense fawn coat with dark markings, long crescent/semi-circular horns, prominent dewlap.',
      milk_yield: '500 kg per lactation',
      fat_percentage: '8.0% - 9.0%',
      uses: 'Sacred Dairy / Pastoral Cultural Heritage',
      climate: 'Adapted exclusively to cold, humid high-altitude montane conditions.',
    },
    Jaffarabadi: {
      native_state: 'Gujarat',
      origin: 'Gir Forest & Coastal Saurashtra',
      characteristics: 'Massive body frame, ultra-broad dome-shaped forehead, flat drooping horns curling up.',
      milk_yield: '2,150 kg per lactation',
      fat_percentage: '7.5% - 8.5%',
      uses: 'High Butterfat Dairy',
      climate: 'Adapted to coastal marshy forest environments with strong foraging power.',
    },
    Murrah: {
      native_state: 'Haryana',
      origin: 'Rohtak, Hisar, Jind Districts',
      characteristics: 'Jet black coat, short tightly curled spiral horns, deep wedge body frame.',
      milk_yield: '2,500 kg per lactation',
      fat_percentage: '7.0% - 7.8%',
      uses: 'Premier Dairy (Black Gold)',
      climate: 'Highly adaptable across all Indian agro-climatic zones.',
    },
  };

  useEffect(() => {
    async function loadBreeds() {
      try {
        const res = await fetchBreedsList();
        setBreeds(res.breeds);
      } catch (err) {
        console.error('Failed to fetch breeds catalog:', err);
      } finally {
        setLoading(false);
      }
    }
    loadBreeds();
  }, []);

  const filteredBreeds = breeds.filter((b) => {
    const matchesFilter = filter === 'all' || b.animal_type.toLowerCase() === filter;
    const matchesSearch =
      b.breed_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      b.origin_region.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            Indigenous Breeds Catalog
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Detailed characteristics, origins, and milk production stats for Indian Cattle & Buffalo breeds.
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex items-center space-x-2 bg-slate-900 p-1.5 rounded-xl border border-slate-800 text-xs">
          <button
            onClick={() => setFilter('all')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition ${
              filter === 'all' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            All (6)
          </button>
          <button
            onClick={() => setFilter('cattle')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition ${
              filter === 'cattle' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            Cattle (3)
          </button>
          <button
            onClick={() => setFilter('buffalo')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition ${
              filter === 'buffalo' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            Buffalo (3)
          </button>
        </div>
      </div>

      {/* Search Input */}
      <div className="relative max-w-md">
        <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-3.5" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search breed name or region..."
          className="w-full bg-slate-900 border border-slate-800 rounded-xl py-2.5 pl-10 pr-4 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition"
        />
      </div>

      {/* Grid of Breed Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredBreeds.map((breed) => {
          const detail = defaultBreedDetails[breed.breed_name] || {};

          return (
            <div
              key={breed.breed_name}
              className="glass-card glass-card-hover p-6 rounded-3xl space-y-4 flex flex-col justify-between"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 uppercase">
                    {breed.animal_type}
                  </span>
                  <span className="text-xs font-bold text-emerald-400 flex items-center space-x-1">
                    <MapPin className="w-3.5 h-3.5" />
                    <span>{detail.native_state || 'India'}</span>
                  </span>
                </div>

                <div>
                  <h3 className="text-2xl font-extrabold text-white">{breed.display_name}</h3>
                  <p className="text-xs text-slate-400 mt-1">{breed.origin_region}</p>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed pt-2 border-t border-slate-800">
                  {breed.description}
                </p>

                {/* Stat Badges */}
                <div className="grid grid-cols-2 gap-2 pt-2 text-xs">
                  <div className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-0.5">
                    <div className="flex items-center space-x-1 text-slate-400 text-[10px]">
                      <Milk className="w-3 h-3 text-indigo-400" />
                      <span>Milk Yield</span>
                    </div>
                    <div className="font-bold text-slate-200">{detail.milk_yield || 'High Yield'}</div>
                  </div>

                  <div className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-0.5">
                    <div className="flex items-center space-x-1 text-slate-400 text-[10px]">
                      <ShieldCheck className="w-3 h-3 text-emerald-400" />
                      <span>Fat Content</span>
                    </div>
                    <div className="font-bold text-emerald-400">{detail.fat_percentage || '4.5% - 8.5%'}</div>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-slate-800/80 text-[11px] text-slate-400">
                <span className="font-semibold text-slate-300">Uses:</span> {detail.uses || 'Dairy'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
